# -*- coding: utf-8 -*-
"""Knockout competition module."""
# pylint: disable=E0611,C0411

from __future__ import annotations
from collections import defaultdict
from typing import List
from loguru import logger

from .competition import Competition
from utils.worker import MixedJudge
from utils.cache import Cache


@Competition.register("knockout")
class Knockout(Competition):
    """
    Knockout competition class.

    Args:
        n (`int`): the number of candidates
        k (`int`): the number of comparisons between each pair

    Total number of comparisons: (n - 1) * k
    """

    def __init__(
        self,
        judge: MixedJudge,
        cache: Cache,
        n: int,
        k: int,
    ):
        """
        Args:
            n (`int`): the number of candidates
            k (`int`): the number of comparisons between each pair
        """
        super().__init__(judge, cache)
        self.n = n
        self.k = k

    def _check_stop(self, candidates: List) -> bool:
        stop = True
        for i in range(1, len(candidates)):
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
        """Run knockout competition.

        Args:
            question (`dict`): the input question
            candidates (`List[dict]`): the input candidates
        """
        candidates = candidates[: self.n]
        knockout_traj = self.cache.load_competition(
            competition_type="knockout",
            instance_id=question["id"],
            category=question["category"],
            suffix=f"{self.n}_{self.k}",
        )
        if knockout_traj:
            return knockout_traj["final"]
        round_num = 0
        knockout_traj = {
            "final": None,
            "detail": {},
        }
        stop_signal = False
        while len(candidates) > 1:
            round_num += 1
            winners = []
            pairs = []
            if not stop_signal:
                stop_signal = self._check_stop(candidates)
            if len(candidates) % 2 == 1:
                winners.append(candidates[-1])
            for i in range(1, len(candidates), 2):
                # pair-wise compare
                if stop_signal:
                    pairs.append(None)
                else:
                    pairs.append(
                        self.judge.pairwise_compare(
                            question=question,
                            candidate_a=candidates[i - 1],
                            candidate_b=candidates[i],
                            k=self.k,
                        ),
                    )
            rounds_detail = []
            for i, pair in enumerate(pairs):
                if stop_signal:
                    rounds_detail.append(
                        {
                            "winner": candidates[i * 2]["cid"],
                            "a": candidates[i * 2]["cid"],
                            "b": candidates[i * 2 + 1]["cid"],
                            "score_a": 0,
                            "score_b": 0,
                        },
                    )
                    winners.append(candidates[i * 2])
                else:
                    pair = pair.result()
                    rounds_detail.append(
                        {
                            "winner": pair["winner"],
                            "a": pair["a"],
                            "b": pair["b"],
                            "score_a": pair["score_a"],
                            "score_b": pair["score_b"],
                        },
                    )
                    if pair["winner"] == candidates[i * 2]["cid"]:
                        winners.append(candidates[i * 2])
                    else:
                        winners.append(candidates[i * 2 + 1])
            knockout_traj["detail"][f"round_{round_num}"] = rounds_detail
            candidates = winners
            logger.info(f"Round {round_num} done")
        knockout_traj["final"] = candidates[0]
        self.cache.save_competition(
            detail=knockout_traj,
            competition_type="knockout",
            instance_id=question["id"],
            category=question["category"],
            suffix=f"{self.n}_{self.k}",
        )
        return candidates[0]

    def default_competition_stats(self) -> dict:
        return defaultdict(
            lambda: {
                "acc": defaultdict(float),
                "majority_acc": defaultdict(float),
                "cnt": 0,
                "details": {},
            },
        )

    def get_question_stats(self, question: dict) -> dict:
        question_stats = {
            "id": question["id"],
        }
        knockout_result = self.cache.load_competition(
            competition_type="knockout",
            instance_id=question["id"],
            category=question["category"],
            suffix=f"{self.n}_{self.k}",
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
        question_stats["majority_acc"] = {
            f"{candidate_num}": question_stats["acc"][f"{candidate_num}"],
        }
        valid_cmp, correct_cmp = self._process_knockout_rounds(
            knockout_result=knockout_result,
            candidates=candidates,
            target=target,
            question_stats=question_stats,
        )
        question_stats["cmp"] = {
            "valid": valid_cmp,
            "correct": correct_cmp,
            "p_cmp": correct_cmp / valid_cmp if valid_cmp > 0 else 0,
        }
        return question_stats

    def update_competition_stats(
        self,
        competition_stats: dict,
        category: str,
        question_stats: dict,
    ) -> None:
        for n in question_stats["acc"]:
            competition_stats[category]["acc"][n] += question_stats["acc"][n]
            competition_stats[category]["majority_acc"][n] += question_stats[
                "majority_acc"
            ][n]
        competition_stats[category]["cnt"] += 1
        competition_stats[category]["details"][
            question_stats["id"]
        ] = question_stats

    def save_competition_stats(self, competition_stats: dict) -> None:
        for category, stats in competition_stats.items():
            for candidate_num in stats["acc"]:
                stats["acc"][candidate_num] /= stats["cnt"]
                stats["majority_acc"][candidate_num] /= stats["cnt"]
            self.cache.save_competition_stats(
                stats=stats,
                category=category,
                competition_type="knockout",
                suffix=f"{self.n}_{self.k}",
            )

    def _process_knockout_rounds(
        self,
        knockout_result: dict,
        candidates: list,
        target: str,
        question_stats: dict,
    ) -> tuple:
        valid_cmp = 0
        correct_cmp = 0
        candidate_num = 1

        for round_num in knockout_result["detail"]:
            candidate_num *= 2
            total = len(knockout_result["detail"][round_num])
            correct = 0

            for pair in knockout_result["detail"][round_num]:
                answer_a = candidates[pair["a"]]["answer"]
                answer_b = candidates[pair["b"]]["answer"]
                answer_winner = candidates[pair["winner"]]["answer"]

                if answer_winner == target:
                    correct += 1

                if (answer_a == target and answer_b != target) or (
                    answer_a != target and answer_b == target
                ):
                    valid_cmp += pair["score_a"] + pair["score_b"]
                    if answer_a == target:
                        correct_cmp += pair["score_a"]
                    if answer_b == target:
                        correct_cmp += pair["score_b"]

            question_stats["acc"][str(candidate_num)] = correct / total
            majority_correct, majority_cnt = self._calculate_majority_accuracy(
                candidates,
                target,
                candidate_num,
            )
            question_stats["majority_acc"][str(candidate_num)] = (
                majority_correct / majority_cnt
            )

        return valid_cmp, correct_cmp

    def _calculate_majority_accuracy(
        self,
        candidates: list,
        target: str,
        candidate_num: int,
    ) -> tuple:
        majority_correct = 0
        majority_cnt = 0

        for i in range(0, self.n, candidate_num):
            majority = self._get_majority(candidates[i : i + candidate_num])
            majority_cnt += 1
            if majority == target:
                majority_correct += 1

        return majority_correct, majority_cnt

    def _get_majority(self, candidates: list) -> str:
        votes = defaultdict(int)
        for c in candidates:
            votes[c["answer"]] += 1
        return max(votes, key=votes.get)  # type: ignore[arg-type]
