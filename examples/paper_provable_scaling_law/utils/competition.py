# -*- coding: utf-8 -*-
"""Competition module."""
from __future__ import annotations
from abc import abstractmethod
from typing import List
from loguru import logger
from agentscope.rpc import async_func, RpcMeta

from .worker import MixedJudge
from .cache import Cache
from .dataset import Dataset


class Competition(metaclass=RpcMeta):
    """A collection of competitions."""

    def __init__(self, judge: MixedJudge, cache: Cache) -> None:
        self.judge = judge
        self.cache = cache

    @async_func
    @abstractmethod
    def competition(
        self,
        question: dict,
        candidates: List[dict],
        **kwargs: dict,
    ) -> dict:
        """Run competition with given method.

        Args:
            question (`dict`): the input question
            candidates (`List[dict]`): the input candidates
            kwargs (`dict`): other arguments
        """

    @abstractmethod
    def calculate_stats(
        self,
        dataset: Dataset,
    ) -> None:
        """Calculate the stats of the competition.

        Args:
            cache (`Cache`): the cache object
            dataset (`Dataset`): the dataset object
        """


class Knockout(Competition):
    def __init__(
        self,
        judge: MixedJudge,
        cache: Cache,
        n: int,
        k: int,
        skip_same: bool = True,
    ):
        """
        Args:
            n (`int`): the number of candidates
            k (`int`): the number of comparisons between each pair
        """
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.skip_same = skip_same

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
        self.cache.load_knockout(
            instance_id=question["id"],
            n=self.n,
            k=self.k,
            category=question["category"],
        )
        round_num = 0
        knockout_traj = {
            "final": None,
            "detail": {},
        }
        all_same = False
        while len(candidates) > 1:
            round_num += 1
            winners = []
            if len(candidates) % 2 == 1:
                winners.append(candidates[-1])
            pairs = []
            if not all_same or self.skip_same:
                seen_answers = set()
                for i in range(len(candidates)):
                    seen_answers.add(candidates[i]["answer"])
                if len(seen_answers) == 1:
                    all_same = True
            for i in range(1, len(candidates), 2):
                # pair-wise compare
                if all_same:
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
                if all_same:
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
        self.cache.save_knockout(
            detail=knockout_traj,
            instance_id=question["id"],
            n=self.n,
            k=self.k,
            category=question["category"],
        )
        return candidates[0]

    def calculate_stats(
        self,
        dataset: Dataset,
    ) -> None:
        """Calculate the stats of the knockout competition."""
        logger.info("Calculating knockout stats...")
        n = self.n
        k = self.k
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
            knockout_result = self.cache.load_knockout(
                instance_id=question["id"],
                n=n,
                k=k,
                category=question["category"],
            )
            candidates = self.cache.load_generation(
                instance_id=question["id"], category=question["category"]
            )[:n]
            target = question["answer"]
            candidate_num = 1
            question_stats["acc"] = {
                f"{candidate_num}": sum(
                    1 for x in candidates if x["answer"] == target
                )
                / len(candidates)
            }
            category_stats[question["category"]]["acc"][
                f"{candidate_num}"
            ] += question_stats["acc"][f"{candidate_num}"]
            valid_cmp = 0
            correct_cmp = 0
            for round_num in knockout_result["detail"]:
                # calculate acc for round x
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
                        valid_cmp += 1
                        if answer_winner == target:
                            correct_cmp += 1
                question_stats["acc"][str(candidate_num)] = correct / total

                if (
                    str(candidate_num)
                    not in category_stats[question["category"]]["acc"]
                ):
                    category_stats[question["category"]]["acc"][
                        str(candidate_num)
                    ] = 0
                category_stats[question["category"]]["acc"][
                    str(candidate_num)
                ] += (correct / total)

            category_stats[question["category"]]["cnt"] += 1
            question_stats["cmp"] = {
                "valid": valid_cmp,
                "correct": correct_cmp,
                "p_cmp": correct_cmp / valid_cmp if valid_cmp > 0 else 0,
            }
            category_stats[question["category"]]["details"][
                str(question["id"])
            ] = question_stats
        for category in category_stats:
            for candidate_num in category_stats[category]["acc"]:
                category_stats[category]["acc"][
                    candidate_num
                ] /= category_stats[category]["cnt"]
            self.cache.save_knockout_stats(
                category_stats[category], n, k, category
            )
        logger.info("Finished calculating knockout stats")


class UCB(Competition):
    def __init__(
        self,
        judge: MixedJudge,
        cache: Cache,
        n: int,
        k: int,
        t: int,
        n_opponent: int,
        c_bonous: float,
    ):
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.t = t
        self.n_opponent = n_opponent
        self.c_bonous = c_bonous

    def competition(
        self,
        question: dict,
        candidates: List[dict],
    ) -> dict:
        """Run ucb competition."""
        import numpy as np

        candidates = candidates[: self.n]
        ucb = np.ones(self.n, dtype=np.float64)
        lcb = np.zeros(self.n, dtype=np.float64)
        avg_win_rate = np.full(self.n, 0.5, dtype=np.float64)
        win_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        lose_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        # whether the candidate is active or not
        active_signal = np.ones(self.n, dtype=np.bool_)
        for t in range(self.t):
            # top_id = np.argmax(ucb + np.random.randn(self.n) * 1e-8 + (active_signal - 1) * 10)
            # find activate candidate id where active_signal == 1
            active_candidate_ids = np.where(active_signal)[0]
            for idx in active_candidate_ids:
                opponent_num = self.n_opponent
                candidate_opponent_list = [
                    x for x in active_candidate_ids if x != idx
                ]
                opponent_list = []
                while opponent_num > len(candidate_opponent_list):
                    opponent_list.extend(candidate_opponent_list)
                    opponent_num -= len(candidate_opponent_list)
                if opponent_num > 0:
                    opponent_list.extend(
                        np.random.choice(
                            candidate_opponent_list,
                            size=opponent_num,
                            replace=False,
                        )
                    )
                futures = []
                for opponent_id in opponent_list:
                    future = self.judge.pairwise_compare(
                        question,
                        candidates[idx]["raw"],
                        candidates[opponent_id]["raw"],
                        k=self.k,
                    )
                    futures.append(future)
                for future in futures:
                    result = future.result()
                    win_cnt_matrix[idx][opponent_id] += result["score_a"]
                    lose_cnt_matrix[idx][opponent_id] += result["score_b"]

            while True:
                for idx in np.where(active_signal)[0]:
                    total_win_count = np.sum(
                        win_cnt_matrix[idx] * active_signal
                    )
                    total_lose_count = np.sum(
                        lose_cnt_matrix[idx] * active_signal
                    )
                    total_count = total_win_count + total_lose_count
                    if total_count >= 1:
                        avg_win_rate[idx] = total_win_count / total_count
                        bonus = np.sqrt(self.c_bonous / total_count)
                        ucb[idx] = min(avg_win_rate[idx] + bonus, 1.0)
                        lcb[idx] = max(avg_win_rate[idx] - bonus, 0.0)
                max_lcb = np.max(lcb * active_signal)
                update_active_signal = False
                for idx in np.where(active_signal)[0]:
                    if ucb[idx] < max_lcb:
                        active_signal[idx] = False
                        update_active_signal = True
                if not update_active_signal:
                    break

            seen_answers = set()
            for idx in np.where(active_signal)[0]:
                if candidates[idx]["answer"] not in seen_answers:
                    seen_answers.add(candidates[idx]["answer"])
            if len(seen_answers) <= 1:
                break


class League(Competition):
    def __init__(self, judge: MixedJudge, cache: Cache, n: int, k: int):
        """
        Args:
            n (`int`): the number of candidates
            k (`int`): the number of comparisons between each pair
        """
        super().__init__(judge, cache)

    def competition(
        self,
        question: dict,
        candidates: List[dict],
    ):
        """Run league competition."""
        # TODO: implement
        pass
