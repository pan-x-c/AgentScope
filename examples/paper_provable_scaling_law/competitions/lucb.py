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


class LUCBState:
    """State of LUCB algorithm."""

    def __init__(
        self,
        cache: Cache,
        qid: str,
        category: str,
        n: int,
        k: int,
        c_bonus: float = 0.99,
        latest_t: int = 0,
    ):
        self.c_bonus = c_bonus
        self.ucb = np.ones(n, dtype=np.float64)
        self.lcb = np.zeros(n, dtype=np.float64)
        self.avg_win_rate = np.full(n, 0.5, dtype=np.float64)
        self.win_cnt_matrix = np.zeros((n, n), dtype=np.float64)
        self.lose_cnt_matrix = np.zeros((n, n), dtype=np.float64)
        # whether the candidate is active or not
        self.active_signal = np.ones(n, dtype=np.bool_)
        self.state_dict = cache.load_competition(
            instance_id=qid,
            competition_type="lucb",
            category=category,
            suffix=f"{n}_{k}_{latest_t}",
        )
        assert (not latest_t) or (
            "detail" in self.state_dict
        ), f"Checkpointed state at round {latest_t} not found."
        if not self.state_dict:
            self.state_dict = {"detail": {}}
        if latest_t != 0:
            round_states = self.state_dict["detail"]
            if len(round_states) > 0:
                latest_round = list(round_states.keys())[-1]
                state = round_states[latest_round]
                self.ucb = np.array(state["ucb"], dtype=np.float64)
                self.lcb = np.array(state["lcb"], dtype=np.float64)
                self.avg_win_rate = np.array(
                    state["avg_win_rate"],
                    dtype=np.float64,
                )
                self.win_cnt_matrix = np.array(
                    state["win_cnt_matrix"],
                    dtype=np.float64,
                )
                self.lose_cnt_matrix = np.array(
                    state["lose_cnt_matrix"],
                    dtype=np.float64,
                )
                self.active_signal = np.zeros(n, dtype=np.bool_)
                self.active_signal[state["active_ids"]] = True

    @property
    def active_ids(self) -> List[int]:
        """Get the active ids."""
        return np.where(self.active_signal)[0].tolist()

    @property
    def max_ucb_id(self) -> int:
        """Get the candidate id with the largest ucb score."""
        return np.argmax(self.ucb * self.active_signal)

    @property
    def max_win_rate_id(self) -> int:
        """Get the candidate id with the largest win rate."""
        return np.argmax(self.avg_win_rate * self.active_signal)

    def add_comparison_result(
        self,
        cid_a: int,
        cid_b: int,
        win_cnt_a: float,
        win_cnt_b: float,
        has_budget: bool = False,
    ) -> None:
        """Add the comparison result to the state."""
        self.win_cnt_matrix[cid_a, cid_b] += win_cnt_a
        self.lose_cnt_matrix[cid_a, cid_b] += win_cnt_b
        if not has_budget:
            self.win_cnt_matrix[cid_b, cid_a] += win_cnt_b
            self.lose_cnt_matrix[cid_b, cid_a] += win_cnt_a

    def update_state(self) -> None:
        """Update the state of the LUCB algorithm."""
        while True:
            for idx in self.active_ids:
                total_win_count = np.sum(
                    self.win_cnt_matrix[idx] * self.active_signal,
                )
                total_lose_count = np.sum(
                    self.lose_cnt_matrix[idx] * self.active_signal,
                )
                total_count = total_win_count + total_lose_count
                if total_count >= 0.5:
                    self.avg_win_rate[idx] = total_win_count / total_count
                    bonus = np.sqrt(self.c_bonus / total_count)
                    self.ucb[idx] = min(self.avg_win_rate[idx] + bonus, 1.0)
                    self.lcb[idx] = max(self.avg_win_rate[idx] - bonus, 0.0)
                else:
                    self.avg_win_rate[idx] = 0.5
                    self.ucb[idx] = 1.0
                    self.lcb[idx] = 0.0
            max_lcb = np.max(self.lcb * self.active_signal)
            update_active_signal = False
            for idx in self.active_ids:
                if self.ucb[idx] < max(max_lcb, 0.5):
                    self.active_signal[idx] = False
                    update_active_signal = True
            if not update_active_signal:
                break

    def add_round_state(
        self,
        round_id: int,
        round_state: dict,
    ) -> None:
        """Record the round state"""
        round_state.update(
            {
                "ucb": self.ucb.tolist(),
                "lcb": self.lcb.tolist(),
                "avg_win_rate": self.avg_win_rate.tolist(),
                "win_cnt_matrix": self.win_cnt_matrix.tolist(),
                "lose_cnt_matrix": self.lose_cnt_matrix.tolist(),
                "active_ids": np.where(self.active_signal)[0].tolist(),
            },
        )
        self.state_dict["detail"][f"round_{round_id + 1}"] = round_state

    def get_state_dict(self) -> dict:
        """Get the LUCB state as a dict."""
        return self.state_dict


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
        latest_t: int = 0,
    ):
        super().__init__(judge, cache)
        self.n = n
        self.k = k
        self.t = t
        self.n_opponent = n_opponent
        self.c_bonus = c_bonus
        self.win_indicator = win_indicator
        self.has_budget = budget != 0
        self.latest_t = latest_t
        if self.has_budget:
            # 0 means no budget
            self.m = budget // (self.k * self.n_opponent)
            assert self.m >= 2 and self.m % 2 == 0
        else:
            self.m = 0

    def get_final(self, state: dict) -> dict:
        """Get the final winner with specific win indicator."""
        if self.win_indicator == "ucb":
            return state["final_ucb"]
        else:
            return state["final_win_rate"]

    def _select_candidates(self, lucb_state: LUCBState) -> List:
        active_candidate_ids = lucb_state.active_ids
        if not self.has_budget:
            # use all active candidates if no budget
            return active_candidate_ids
        m_top_ucb = self.m // 2
        m_bottom_lcb = self.m - m_top_ucb
        if len(active_candidate_ids) > m_top_ucb:
            scores_for_top_ucb = (
                lucb_state.ucb
                + np.random.randn(self.n) * 1e-8
                + lucb_state.active_signal * 10
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
                lucb_state.lcb
                + np.random.randn(self.n) * 1e-8
                - lucb_state.active_signal * 10
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
        comparison_ids: List,
        candidates: List,
        question: dict,
    ) -> List:
        futures = []
        for idx in comparison_ids:
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
        stop = True
        for idx in active_ids:
            if (
                candidates[idx]["answer"]
                != candidates[active_ids[0]]["answer"]
            ):
                stop = False
                break
        return stop

    def run_lucb(self, question: dict, candidates: List) -> Tuple:
        """The main procedure of LUCB"""
        lucb_state = LUCBState(
            cache=self.cache,
            qid=question["id"],
            category=question["category"],
            n=self.n,
            k=self.k,
            latest_t=self.latest_t,
        )
        total_cmp_cnt = 0
        for t in range(self.latest_t, self.t):
            active_ids = lucb_state.active_ids
            if self._check_stop(candidates=candidates, active_ids=active_ids):
                break
            round_state = {
                "compare_cnt": 0,
                "active_ids": [],
                "comparisons": [],
            }
            comparision_ids = self._select_candidates(lucb_state=lucb_state)
            futures = self._run_lucb_round(
                activate_ids=active_ids,
                comparison_ids=comparision_ids,
                candidates=candidates,
                question=question,
            )
            for future in futures:
                result = future.result()
                total_cmp_cnt += self.k
                round_state["compare_cnt"] += self.k  # type: ignore[operator]
                round_state["comparisons"].append(result)
                lucb_state.add_comparison_result(
                    cid_a=result["a"],
                    cid_b=result["b"],
                    win_cnt_a=result["score_a"],
                    win_cnt_b=result["score_b"],
                    has_budget=self.has_budget,
                )
            lucb_state.update_state()
            lucb_state.add_round_state(round_id=t, round_state=round_state)

        state = lucb_state.get_state_dict()
        state["final_ucb"] = candidates[lucb_state.max_ucb_id]
        state["final_win_rate"] = candidates[lucb_state.max_win_rate_id]
        state["total_cmp_cnt"] = state.get("total_cmp_cnt", 0) + total_cmp_cnt
        final = self.get_final(state)
        state["final"] = final
        return final, state

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
                active_signal[active_ids] = True
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
