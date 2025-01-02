# -*- coding: utf-8 -*-
"""League competition module."""
# pylint: disable=E0611,C0411
from __future__ import annotations
from typing import List
from collections import defaultdict

from .competition import Competition
from utils.worker import MixedJudge
from utils.cache import Cache


@Competition.register("league")
class League(Competition):
    """
    League Competition class.

    Args:
        n (`int`): the number of candidates
        k (`int`): the number of comparisons between each pair
        m (`int`): the number of competitors for each candidate, default to n-1

    Total number of pair-wise comparisons: `n * k * m`
    """

    def __init__(
        self,
        judge: MixedJudge,
        cache: Cache,
        n: int,
        k: int,
        m: int = None,
    ) -> None:
        """
        Init a League Competition instance.

        Args:
            judge (`MixedJudge`): the judge instance
            cache (`Cache`): the cache instance
            n (`int`): the number of candidates
            k (`int`): the number of comparisons between each pair
            m (`int`): the number of competitors for each candidate,
                default to n-1
        """
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.m = m if m else (n - 1)
        assert self.m < self.n and self.m > 0

    def _check_stop(self, candidates: List[dict]) -> bool:
        stop = True
        for i in range(1, self.n):
            if candidates[i]["answer"] != candidates[0]["answer"]:
                stop = False
                break
        return stop

    def competition(
        self,
        question: dict,
        candidates: List[dict],
        **kwargs: dict,
    ) -> dict:
        """Run league competition

        Args:
            question (`dict`): the input question
            candidates (`List[dict]`): the input candidates
        """
        candidates = candidates[: self.n]
        league_record = self.cache.load_competition(
            instance_id=question["id"],
            competition_type="league",
            category=question["category"],
            suffix=f"{self.n}_{self.k}_{self.m}",
        )
        league_record = {"final": None}
        if league_record:
            return league_record["final"]
        cmp_matrix = [[0 for _ in range(self.n)] for i in range(self.n)]
        if self._check_stop(candidates):
            final = candidates[0]
            league_record = {
                "final": final,
                "score_board": [0 for _ in range(self.n)],
                "score_matrix": [
                    [0 for _ in range(self.n)] for i in range(self.n)
                ],
            }
        else:
            for i in range(self.n):
                for j in range(1, self.m + 1):
                    opp = (i + j) % self.n
                    if i < opp:
                        cmp_matrix[i][opp] += self.k
                    else:
                        cmp_matrix[opp][i] += self.k
            pairs = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if cmp_matrix[i][j] == 0:
                        continue
                    pairs.append(
                        self.judge.pairwise_compare(
                            question=question,
                            candidate_a=candidates[i],
                            candidate_b=candidates[j],
                            k=cmp_matrix[i][j],
                        ),
                    )
            score_board = [0 for _ in range(self.n)]
            score_matrix = [[0 for _ in range(self.n)] for i in range(self.n)]
            for pair in pairs:
                pair = pair.result()
                a_id = pair["a"]
                b_id = pair["b"]
                score_board[a_id] += pair["score_a"]
                score_board[b_id] += pair["score_b"]
                score_matrix[a_id][b_id] += pair["score_a"]
                score_matrix[b_id][a_id] += pair["score_b"]
            final = candidates[score_board.index(max(score_board))]
            league_record["final"] = final
            league_record["score_board"] = score_board
            league_record["score_matrix"] = score_matrix
        self.cache.save_competition(
            detail=league_record,
            competition_type="league",
            instance_id=question["id"],
            category=question["category"],
            suffix=f"{self.n}_{self.k}_{self.m}",
        )
        return final

    def default_competition_stats(self) -> dict:
        return defaultdict(
            lambda: {
                "acc": defaultdict(float),
                "acc_vs_m": defaultdict(float),
                "cnt": 0,
                "details": {},
            },
        )

    def get_question_stats(self, question: dict) -> dict:
        import numpy as np

        qid = question["id"]
        category = question["category"]
        competition_result = self.cache.load_competition(
            instance_id=qid,
            competition_type="league",
            category=category,
            suffix=f"{self.n}_{self.k}_{self.m}",
        )
        score_matrix = np.array(
            competition_result["score_matrix"],
            dtype=np.float64,
        )

        candidates = self.cache.load_generation(
            instance_id=qid,
            category=category,
        )[: self.n]
        target = question["answer"]
        candidate_num = 1
        question_stats = {"id": question["id"]}
        question_stats["acc"] = {
            "1": sum(1 for x in candidates if x["answer"] == target)
            / len(candidates),
        }

        # calculate acc vs n
        while candidate_num < self.n:
            candidate_num += 1
            question_stats["acc"][f"{candidate_num}"] = 0
            for i in range(0, self.n):
                indices = [(i + j) % self.n for j in range(candidate_num)]
                sub_matrix = score_matrix[np.ix_(indices, indices)]
                sub_board = [sum(sub_matrix[j]) for j in range(candidate_num)]
                sub_final = candidates[indices[np.argmax(sub_board)]]
                question_stats["acc"][f"{candidate_num}"] += int(
                    sub_final["answer"] == target,
                )
            question_stats["acc"][f"{candidate_num}"] /= self.n
        valid_cmp = 0
        correct_cmp = 0
        # calculate_p_cmp
        for i in range(self.n):
            i_correct = candidates[i]["answer"] == target
            for j in range(self.n):
                j_correct = candidates[j]["answer"] == target
                if i_correct != j_correct:
                    valid_cmp += score_matrix[i][j]
                    if i_correct:
                        correct_cmp += score_matrix[i][j]
        question_stats["cmp"] = {
            "valid": valid_cmp,
            "correct": correct_cmp,
            "p_cmp": correct_cmp / valid_cmp if valid_cmp > 0 else 0,
        }

        # calculate acc vs m
        question_stats["acc_vs_m"] = {}
        for m in range(1, self.m + 1):
            score_board = []
            for i in range(self.n):
                score_board.append(
                    sum(
                        score_matrix[i][(i + j) % self.n]
                        for j in range(1, m + 1)
                    ),
                )
            max_score = np.max(score_board)
            finals = [
                candidates[i]
                for i, score in enumerate(score_board)
                if score == max_score
            ]
            question_stats["acc_vs_m"][str(m)] = sum(
                1 for final in finals if final["answer"] == target
            ) / len(finals)
        return question_stats

    def update_competition_stats(
        self,
        competition_stats: dict,
        category: str,
        question_stats: dict,
    ) -> None:
        for n in question_stats["acc"].keys():
            competition_stats[category]["acc"][n] += question_stats["acc"][n]

        for m in question_stats["acc_vs_m"].keys():
            competition_stats[category]["acc_vs_m"][m] += question_stats[
                "acc_vs_m"
            ][m]
        competition_stats[category]["cnt"] += 1
        competition_stats[category]["details"][
            question_stats["id"]
        ] = question_stats

    def save_competition_stats(self, competition_stats: dict) -> None:
        for category, stats in competition_stats.items():
            for n in stats["acc"]:
                stats["acc"][n] /= stats["cnt"]
            for m in stats["acc_vs_m"]:
                stats["acc_vs_m"][m] /= stats["cnt"]
            self.cache.save_competition_stats(
                stats=stats,
                competition_type="league",
                category=category,
                suffix=f"{self.n}_{self.k}_{self.m}",
            )
