# -*- coding: utf-8 -*-
"""UCB competition module."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .competition import Competition

from ..utils.worker import MixedJudge
from ..utils.cache import Cache
from ..utils.dataset import Dataset


@Competition.register("lucb")
class LUCB(Competition):
    """An implementation of LUCB algorithm."""

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
        budget: int = 0,
    ):
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.t = t
        self.n_opponent = n_opponent
        self.c_bonus = c_bonus
        self.win_indicator = win_indicator
        self.has_budget = budget != 0
        if self.has_budget:
            # 0 means no budget
            self.m = budget // (self.k * self.n_opponent)
            assert self.m >= 2 and self.m % 2 == 0
        else:
            self.m = 0

    def get_final(self, stats: dict) -> dict:
        """Get the final winner with specific win indicator."""
        if self.win_indicator == "ucb":
            return stats["final_ucb"]
        else:
            return stats["final_win_rate"]

    def _select_candidates_for_comparison(
        self,
        active_signal: np.ndarray,
        ucb: np.ndarray,
        lcb: np.ndarray,
    ) -> List:
        active_candidate_ids = np.where(active_signal)[0]
        if not self.has_budget:
            # use all active candidates if no budget
            return active_candidate_ids.to_list()
        m_top_ucb = self.m // 2
        m_bottom_lcb = self.m - m_top_ucb
        if len(active_candidate_ids) > m_top_ucb:
            scores_for_top_ucb = (
                ucb + np.random.randn(self.n) * 1e-8 + active_signal * 10
            )
            sorted_idx = np.argsort(scores_for_top_ucb)
            idx_top_ucb = sorted_idx[-m_top_ucb:].tolist()
        else:
            idx_top_ucb = np.random.choice(
                active_candidate_ids,
                m_top_ucb,
                replace=True,
            ).tolist()
        if len(active_candidate_ids) > m_bottom_lcb:
            scores_for_bottom_lcb = (
                lcb + np.random.randn(self.n) * 1e-8 - active_signal * 10
            )
            sorted_idx = np.argsort(scores_for_bottom_lcb)
            idx_bottom_lcb = sorted_idx[:m_bottom_lcb].tolist()
        else:
            idx_bottom_lcb = np.random.choice(
                active_candidate_ids,
                m_bottom_lcb,
                replace=True,
            ).tolist()
        return idx_top_ucb + idx_bottom_lcb

    def run_lucb(self, question: dict, candidates: List) -> Tuple:
        """The main procedure of LUCB"""
        total_cmp_cnt = 0
        ucb_stats = {"final": None, "detail": {}}
        ucb = np.ones(self.n, dtype=np.float64)
        lcb = np.zeros(self.n, dtype=np.float64)
        avg_win_rate = np.full(self.n, 0.5, dtype=np.float64)
        win_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        lose_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        # whether the candidate is active or not
        active_signal = np.ones(self.n, dtype=np.bool_)
        for t in range(self.t):
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
            # find active candidate id where active_signal == 1
            candidates_for_comparision = (
                self._select_candidates_for_comparison(
                    active_signal=active_signal, ucb=ucb, lcb=lcb
                )
            )
            active_candidate_ids = np.where(active_signal)[0]
            futures = []
            for idx in candidates_for_comparision:
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
                        ),
                    )
                for opponent_id in opponent_list:
                    futures.append(
                        self.judge.pairwise_compare(
                            question,
                            candidates[idx],
                            candidates[opponent_id],
                            k=self.k,
                            reuse=False,
                        ),
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
                        win_cnt_matrix[idx] * active_signal,
                    )
                    total_lose_count = np.sum(
                        lose_cnt_matrix[idx] * active_signal,
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
                },
            )
            ucb_stats["detail"][f"round_{t + 1}"] = round_stats

        max_ucb_idx = np.argmax(ucb * active_signal)
        max_win_rate_idx = np.argmax(avg_win_rate * active_signal)
        ucb_stats["final_ucb"] = candidates[max_ucb_idx]
        ucb_stats["final_win_rate"] = candidates[max_win_rate_idx]
        ucb_stats["total_cmp_cnt"] = total_cmp_cnt
        final = self.get_final(ucb_stats)
        ucb_stats["final"] = final
        return final, ucb_stats

    def competition(
        self,
        question: dict,
        candidates: List[dict],
        **kwargs: dict,
    ) -> dict:
        """Run lucb competition."""
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
        final, ucb_stats = self.run_lucb(
            question=question, candidates=candidates
        )
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
            final_ids = np.array(range(n))
            question_stats["acc"] = {
                "avg": sum(1 for x in candidates if x["answer"] == target)
                / len(candidates),
            }
            question_stats["acc"]["0"] = question_stats["acc"]["avg"]
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
                                ],
                            )
                            * active_signal
                        )
                    else:
                        scores = (
                            np.array(
                                ucb_result["detail"][f"round_{round_num}"][
                                    "avg_win_rate"
                                ],
                            )
                            * active_signal
                        )
                    max_score = np.max(scores)
                    final_ids = np.where(
                        np.isclose(scores, max_score, atol=1e-8),
                    )[0]
                if (
                    str(round_num)
                    not in category_stats[question["category"]]["acc"]
                ):
                    category_stats[question["category"]]["acc"][
                        str(round_num)
                    ] = 0
                question_stats["acc"][str(round_num)] = sum(
                    int(candidates[final_idx]["answer"] == target)
                    for final_idx in final_ids
                ) / len(final_ids)
                category_stats[question["category"]]["acc"][
                    str(round_num)
                ] += question_stats["acc"][str(round_num)]
            category_stats[question["category"]]["cnt"] += 1
            question_stats["cmp"] = {
                "valid": valid_cmp,
                "correct": correct_cmp,
                "p_cmp": correct_cmp / valid_cmp if valid_cmp > 0 else 0,
            }
            category_stats[question["category"]]["details"][
                str(question["id"])
            ] = question_stats
        for category, stats in category_stats.items():
            for t in stats["acc"]:
                stats["acc"][t] /= stats["cnt"]
            self.cache.save_ucb_stats(stats, n, k, t, category)
