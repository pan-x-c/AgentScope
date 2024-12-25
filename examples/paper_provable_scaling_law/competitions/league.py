# -*- coding: utf-8 -*-
"""League competition module."""
# pylint: disable=E0611,C0411
from __future__ import annotations
from typing import List
from loguru import logger

from .competition import Competition
from utils.worker import MixedJudge
from utils.cache import Cache
from utils.dataset import Dataset


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
        if league_record:
            return league_record["final"]
        else:
            league_record = {
                "final": None,
            }
        cmp_matrix = [[0 for _ in range(self.n)] for i in range(self.n)]
        all_same = True
        for i in range(1, self.n):
            if candidates[i] != candidates[0]:
                all_same = False
                break
        if all_same:
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

    def calculate_stats(
        self,
        dataset: Dataset,
    ) -> None:
        import numpy as np

        logger.info("Calculating league stats ...")
        category_stats = {}
        for question in dataset:
            if question["category"] not in category_stats:
                category_stats[question["category"]] = {
                    "acc": {
                        "1": 0,
                    },
                    "cnt": 0,
                    "details": {},
                }
            question_stats = {}
            competition_result = self.cache.load_competition(
                instance_id=question["id"],
                competition_type="league",
                category=question["category"],
                suffix=f"{self.n}_{self.k}_{self.m}",
            )
            candidates = self.cache.load_generation(
                instance_id=question["id"],
                category=question["category"],
            )[: self.n]
            target = question["answer"]
            candidate_num = 1
            question_stats["acc"] = {
                f"{candidate_num}": sum(
                    1 for x in candidates if x["answer"] == target
                )
                / len(candidates),
            }
            category_stats[question["category"]]["acc"][
                f"{candidate_num}"
            ] += question_stats["acc"][f"{candidate_num}"]
            score_matrix = np.array(
                competition_result["score_matrix"],
                dtype=np.float64,
            )
            # calculate acc
            while candidate_num < self.n:
                candidate_num += 1
                question_stats["acc"][f"{candidate_num}"] = 0
                for i in range(0, self.n):
                    indices = [(i + j) % self.n for j in range(candidate_num)]
                    sub_matrix = score_matrix[np.ix_(indices, indices)]
                    sub_board = [
                        sum(sub_matrix[j]) for j in range(candidate_num)
                    ]
                    sub_final = candidates[indices[np.argmax(sub_board)]]
                    question_stats["acc"][f"{candidate_num}"] += int(
                        sub_final["answer"] == target,
                    )
                question_stats["acc"][f"{candidate_num}"] /= self.n
                if (
                    str(candidate_num)
                    not in category_stats[question["category"]]["acc"]
                ):
                    category_stats[question["category"]]["acc"][
                        str(candidate_num)
                    ] = 0
                category_stats[question["category"]]["acc"][
                    str(candidate_num)
                ] += question_stats["acc"][f"{candidate_num}"]
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
            category_stats[question["category"]]["cnt"] += 1
            category_stats[question["category"]]["details"][
                str(question["id"])
            ] = question_stats
        for category in category_stats:
            for candidate_num in category_stats[category]["acc"]:
                category_stats[category]["acc"][
                    candidate_num
                ] /= category_stats[category]["cnt"]
            self.cache.save_competition_stats(
                stats=category_stats[category],
                competition_type="league",
                category=category,
                suffix=f"{self.n}_{self.k}_{self.m}",
            )
        logger.info("Finished calculating league stats")
