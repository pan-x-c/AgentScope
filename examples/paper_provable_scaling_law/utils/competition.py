# -*- coding: utf-8 -*-
"""Competition module."""
from __future__ import annotations
from abc import abstractmethod
from tqdm import tqdm
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
        knockout_traj = self.cache.load_knockout(
            instance_id=question["id"],
            n=self.n,
            k=self.k,
            category=question["category"],
        )
        if knockout_traj:
            return knockout_traj["final"]
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
        n_opponent: int = 2,
        c_bonus: float = 1.0,
        win_indicator: str = "win_rate",
    ):
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.t = t
        self.n_opponent = n_opponent
        self.c_bonus = c_bonus
        self.win_indicator = win_indicator

    def get_final(self, stats: dict) -> dict:
        if self.win_indicator == "ucb":
            return stats["final_ucb"]
        else:
            return stats["final_win_rate"]

    def competition(
        self,
        question: dict,
        candidates: List[dict],
    ) -> dict:
        """Run ucb competition."""
        import numpy as np

        candidates = candidates[: self.n]

        ucb_stats = self.cache.load_ucb(
            instance_id=question["id"],
            n=self.n,
            k=self.k,
            t=self.t,
            category=question["category"],
        )
        if ucb_stats:
            return self.get_final(ucb_stats)
        total_cmp_cnt = 0
        ucb_stats = {"final": None, "detail": {}}
        ucb = np.ones(self.n, dtype=np.float64)
        lcb = np.zeros(self.n, dtype=np.float64)
        avg_win_rate = np.full(self.n, 0.5, dtype=np.float64)
        win_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        lose_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        # whether the candidate is active or not
        active_signal = np.ones(self.n, dtype=np.bool_)
        for t in tqdm(range(self.t)):
            seen_answers = set()
            for idx in np.where(active_signal)[0]:
                if candidates[idx]["answer"] not in seen_answers:
                    seen_answers.add(candidates[idx]["answer"])
            if len(seen_answers) <= 1:
                break
            round_stats = {
                "compare_cnt": 0,
                "active_ids": [],
                "comparisons": [],
            }
            # find activate candidate id where active_signal == 1
            active_candidate_ids = np.where(active_signal)[0]
            futures = []
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
                for opponent_id in opponent_list:
                    futures.append(
                        self.judge.pairwise_compare(
                            question,
                            candidates[idx],
                            candidates[opponent_id],
                            k=self.k,
                            reuse=False,
                        )
                    )
            for future in futures:
                result = future.result()
                total_cmp_cnt += self.k
                round_stats["compare_cnt"] += self.k
                round_stats["comparisons"].append(result)
                win_cnt_matrix[result["a"]][result["b"]] += result["score_a"]
                lose_cnt_matrix[result["a"]][result["b"]] += result["score_b"]
                win_cnt_matrix[result["b"]][result["a"]] += result["score_b"]
                lose_cnt_matrix[result["b"]][result["a"]] += result["score_a"]

            while True:
                for idx in np.where(active_signal)[0]:
                    total_win_count = np.sum(
                        win_cnt_matrix[idx] * active_signal
                    )
                    total_lose_count = np.sum(
                        lose_cnt_matrix[idx] * active_signal
                    )
                    total_count = total_win_count + total_lose_count
                    if total_count >= 0.5:
                        avg_win_rate[idx] = total_win_count / total_count
                        bonus = np.sqrt(self.c_bonus / total_count)
                        ucb[idx] = min(avg_win_rate[idx] + bonus, 1.0)
                        lcb[idx] = max(avg_win_rate[idx] - bonus, 0.0)
                    else:
                        avg_win_rate[idx] = 0.5
                        ucb[idx] = 1.0
                        lcb[idx] = 0.0
                max_lcb = np.max(lcb * active_signal)
                update_active_signal = False
                for idx in np.where(active_signal)[0]:
                    if ucb[idx] < max_lcb:
                        active_signal[idx] = False
                        update_active_signal = True
                if not update_active_signal:
                    break
            round_stats.update(
                {
                    "active_ids": np.where(active_signal)[0].tolist(),
                    "ucb": ucb.tolist(),
                    "lcb": lcb.tolist(),
                    "avg_win_rate": avg_win_rate.tolist(),
                    "win_cnt_matrix": win_cnt_matrix.tolist(),
                    "lose_cnt_matrix": lose_cnt_matrix.tolist(),
                }
            )
            ucb_stats["detail"][f"round_{t + 1}"] = round_stats

        max_ucb_idx = np.argmax(ucb * active_signal)
        max_win_rate_idx = np.argmax(avg_win_rate * active_signal)
        ucb_stats["final_ucb"] = candidates[max_ucb_idx]
        ucb_stats["final_win_rate"] = candidates[max_win_rate_idx]
        ucb_stats["total_cmp_cnt"] = total_cmp_cnt
        final = self.get_final(ucb_stats)
        ucb_stats["final"] = final
        self.cache.save_ucb(
            detail=ucb_stats,
            instance_id=question["id"],
            n=self.n,
            k=self.k,
            t=self.t,
            category=question["category"],
        )
        return final

    def calculate_stats(self, dataset: Dataset):
        import numpy as np

        n = self.n
        k = self.k
        t = self.t
        category_stats = {}
        for question in dataset:
            if question["category"] not in category_stats:
                category_stats[question["category"]] = {
                    "acc": {
                        "0": 0,
                    },
                    "cnt": 0,
                    "details": {},
                }
            question_stats = {}
            ucb_result = self.cache.load_ucb(
                instance_id=question["id"],
                n=n,
                k=k,
                t=t,
                category=question["category"],
            )
            candidates = self.cache.load_generation(
                instance_id=question["id"],
                category=question["category"],
            )[:n]
            target = question["answer"]
            final_idx = 0
            question_stats["acc"] = {
                "avg": sum(1 for x in candidates if x["answer"] == target)
                / len(candidates),
                "0": int(candidates[final_idx]["answer"] == target),
            }
            category_stats[question["category"]]["acc"]["0"] += question_stats[
                "acc"
            ]["0"]
            valid_cmp = 0
            correct_cmp = 0
            for round_num in range(1, t + 1):
                if f"round_{round_num}" in ucb_result["detail"]:
                    # this round has been calculated
                    for pair in ucb_result["detail"][f"round_{round_num}"][
                        "comparisons"
                    ]:
                        answer_a = candidates[pair["a"]]["answer"]
                        answer_b = candidates[pair["b"]]["answer"]
                        answer_winner = candidates[pair["winner"]]["answer"]
                        if (answer_a == target and answer_b != target) or (
                            answer_a != target and answer_b == target
                        ):
                            valid_cmp += 1
                            if answer_winner == target:
                                correct_cmp += 1
                    active_signal = np.zeros(n, dtype=np.bool_)
                    for idx in ucb_result["detail"][f"round_{round_num}"][
                        "active_ids"
                    ]:
                        active_signal[idx] = True
                    if self.win_indicator == "ucb":
                        scores = (
                            np.array(
                                ucb_result["detail"][f"round_{round_num}"][
                                    "ucb"
                                ]
                            )
                            * active_signal
                        )
                    else:
                        scores = (
                            np.array(
                                ucb_result["detail"][f"round_{round_num}"][
                                    "avg_win_rate"
                                ]
                            )
                            * active_signal
                        )
                    final_idx = np.argmax(scores)
                if (
                    str(round_num)
                    not in category_stats[question["category"]]["acc"]
                ):
                    category_stats[question["category"]]["acc"][
                        str(round_num)
                    ] = 0
                category_stats[question["category"]]["acc"][
                    str(round_num)
                ] += int(candidates[final_idx]["answer"] == target)
                question_stats["acc"][str(round_num)] = int(
                    candidates[final_idx]["answer"] == target
                )

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
            for t in category_stats[category]["acc"]:
                category_stats[category]["acc"][t] /= category_stats[category][
                    "cnt"
                ]
            self.cache.save_ucb_stats(
                category_stats[category], n, k, t, category
            )


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
