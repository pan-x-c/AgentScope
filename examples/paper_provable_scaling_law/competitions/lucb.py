# -*- coding: utf-8 -*-
"""LUCB competition module."""
# pylint: disable=E0611,C0411

from __future__ import annotations
from typing import List, Tuple
from collections import defaultdict
import numpy as np

from .competition import Competition

from utils.worker import MixedJudge
from utils.cache import Cache


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
        n_opponent: int = 1,
        c_bonus: float = 0.99,
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
            return active_candidate_ids.tolist()
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

    def _run_lucb_round(
        self,
        activate_ids: List,
        candidates: List,
        question: dict,
    ) -> List:
        futures = []
        for idx in candidates:
            opponent_num = self.n_opponent
            candidate_opponent_list = [x for x in activate_ids if x != idx]
            opponent_list = []
            while opponent_num >= len(candidate_opponent_list):
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
        return futures

    def _check_stop(self, candidates: List, active_ids: List) -> bool:
        """Check if the stop condition is satisfied."""
        seen_answers = set()
        for idx in active_ids:
            if candidates[idx]["answer"] not in seen_answers:
                seen_answers.add(candidates[idx]["answer"])
        return len(seen_answers) <= 1

    def run_lucb(self, question: dict, candidates: List) -> Tuple:
        """The main procedure of LUCB"""
        total_cmp_cnt = 0
        lucb_stats = {"detail": {}}
        ucb = np.ones(self.n, dtype=np.float64)
        lcb = np.zeros(self.n, dtype=np.float64)
        avg_win_rate = np.full(self.n, 0.5, dtype=np.float64)
        win_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        lose_cnt_matrix = np.zeros((self.n, self.n), dtype=np.float64)
        # whether the candidate is active or not
        active_signal = np.ones(self.n, dtype=np.bool_)
        for t in range(self.t):
            active_ids = np.where(active_signal)[0]
            if self._check_stop(candidates=candidates, active_ids=active_ids):
                break
            round_stats = {
                "compare_cnt": 0,
                "active_ids": [],
                "comparisons": [],
            }
            # find active candidate id where active_signal == 1
            candidates_for_comparision = (
                self._select_candidates_for_comparison(
                    active_signal=active_signal,
                    ucb=ucb,
                    lcb=lcb,
                )
            )
            futures = self._run_lucb_round(
                activate_ids=active_ids,
                candidates=candidates_for_comparision,
                question=question,
            )
            for future in futures:
                result = future.result()
                total_cmp_cnt += self.k
                round_stats["compare_cnt"] += self.k  # type: ignore[operator]
                round_stats["comparisons"].append(result)
                win_cnt_matrix[result["a"]][result["b"]] += result["score_a"]
                lose_cnt_matrix[result["a"]][result["b"]] += result["score_b"]
                if not self.has_budget:
                    win_cnt_matrix[result["b"]][result["a"]] += result[
                        "score_b"
                    ]
                    lose_cnt_matrix[result["b"]][result["a"]] += result[
                        "score_a"
                    ]

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
                    if ucb[idx] < max(max_lcb, 0.5):
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
            lucb_stats["detail"][f"round_{t + 1}"] = round_stats

        lucb_stats["total_cmp_cnt"] = total_cmp_cnt
        final = self.get_final(lucb_stats)
        lucb_stats["final"] = final
        return final, lucb_stats

    def competition(
        self,
        question: dict,
        candidates: List[dict],
        **kwargs: dict,
    ) -> dict:
        """Run lucb competition."""
        candidates = candidates[: self.n]

        lucb_stats = self.cache.load_competition(
            instance_id=question["id"],
            competition_type="lucb",
            category=question["category"],
            suffix=f"{self.n}_{self.k}_{self.t}",
        )
        if lucb_stats:
            return self.get_final(lucb_stats)
        final, lucb_stats = self.run_lucb(
            question=question,
            candidates=candidates,
        )
        self.cache.save_competition(
            detail=lucb_stats,
            competition_type="lucb",
            instance_id=question["id"],
            category=question["category"],
            suffix=f"{self.n}_{self.k}_{self.t}",
        )
        return final

    def default_competition_stats(self) -> dict:
        return defaultdict(
            lambda: {
                "acc": defaultdict(float),
                "pool_acc": defaultdict(float),
                "pool_size": defaultdict(float),
                "cnt": 0,
                "details": {},
            },
        )

    def get_question_stats(self, question: dict) -> dict:
        ucb_result = self.cache.load_competition(
            instance_id=question["id"],
            competition_type="lucb",
            category=question["category"],
            suffix=f"{self.n}_{self.k}_{self.t}",
        )
        candidates = self.cache.load_generation(
            instance_id=question["id"],
            category=question["category"],
        )[: self.n]
        target = question["answer"]
        question_stats = {"id": question["id"]}
        question_stats["acc"] = {
            "0": sum(1 for x in candidates if x["answer"] == target)
            / len(candidates),
        }
        question_stats["pool_acc"] = {"0": question_stats["acc"]["0"]}
        question_stats["pool_size"] = {"0": self.n}
        valid_cmp, correct_cmp = self._process_lucb_rounds(
            ucb_result=ucb_result,
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
        for t in question_stats["acc"]:
            competition_stats[category]["acc"][t] += question_stats["acc"][t]
            competition_stats[category]["pool_size"][t] += question_stats[
                "pool_size"
            ][t]
            competition_stats[category]["pool_acc"][t] += question_stats[
                "pool_acc"
            ][t]
        competition_stats[category]["cnt"] += 1
        competition_stats[category]["details"][
            question_stats["id"]
        ] = question_stats

    def save_competition_stats(self, competition_stats: dict) -> None:
        for category, stats in competition_stats.items():
            for t in stats["acc"]:
                stats["acc"][t] /= stats["cnt"]
                stats["pool_size"][t] /= stats["cnt"]
                stats["pool_acc"][t] /= stats["cnt"]
            self.cache.save_competition_stats(
                stats=stats,
                competition_type="lucb",
                category=category,
                suffix=f"{self.n}_{self.k}_{self.t}",
            )

    def _process_lucb_rounds(
        self,
        ucb_result: dict,
        candidates: list,
        target: str,
        question_stats: dict,
    ) -> tuple:
        valid_cmp = 0
        correct_cmp = 0
        active_ids = range(self.n)
        final_ids = range(self.n)
        pool_size = self.n
        for round_num in range(1, self.t + 1):
            round_name = f"round_{round_num}"
            if round_name in ucb_result["detail"]:
                # this round has been calculated
                for pair in ucb_result["detail"][round_name]["comparisons"]:
                    answer_a = candidates[pair["a"]]["answer"]
                    answer_b = candidates[pair["b"]]["answer"]
                    if (answer_a == target and answer_b != target) or (
                        answer_a != target and answer_b == target
                    ):
                        valid_cmp += pair["cmp_num"]
                        if answer_a == target:
                            correct_cmp += pair["score_a"]
                        if answer_b == target:
                            correct_cmp += pair["score_b"]
                active_ids = ucb_result["detail"][round_name]["active_ids"]
                active_signal = np.zeros(self.n, dtype=np.bool_)
                for idx in active_ids:
                    active_signal[idx] = True
                pool_size = len(active_ids)
                scores = (
                    np.array(
                        ucb_result["detail"][round_name][self.win_indicator],
                    )
                    * active_signal
                )
                max_score = np.max(scores)
                final_ids = np.where(
                    np.isclose(scores, max_score, atol=1e-8),
                )[0].tolist()
            else:
                pool_size = 1
                active_ids = [active_ids[0]]
            question_stats["acc"][str(round_num)] = sum(
                int(candidates[final_idx]["answer"] == target)
                for final_idx in final_ids
            ) / len(final_ids)
            question_stats["pool_acc"][str(round_num)] = sum(
                int(candidates[idx]["answer"] == target) for idx in active_ids
            ) / len(active_ids)
            question_stats["pool_size"][str(round_num)] = pool_size
        return valid_cmp, correct_cmp
