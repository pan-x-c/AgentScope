# -*- coding: utf-8 -*-
"""Competition module."""
from tqdm import tqdm
from typing import List
from loguru import logger
from agentscope.rpc import async_func, RpcMeta

from utils.worker import MixedJudge
from utils.cache import Cache


class Competition(metaclass=RpcMeta):
    """A collection of competitions."""

    def __init__(self, judge: MixedJudge, cache: Cache) -> None:
        self.judge = judge
        self.cache = cache

    @async_func
    def competition(
        self,
        method: str,
        question: dict,
        candidates: List[dict],
        **kwargs: dict,
    ) -> dict:
        """Run competition with given method.

        Args:
            method (`str`): competition method, e.g. "knockout"
            question (`dict`): the input question
            candidates (`List[dict]`): the input candidates
        """
        if method == "knockout":
            return self.knockout(question, candidates, **kwargs)
        elif method == "league":
            return self.league(question, candidates, **kwargs)
        else:
            raise NotImplementedError

    def knockout(
        self,
        question: dict,
        candidates: List[dict],
        candidate_num: int,
        k: int,
    ) -> dict:
        """Run knockout competition.

        Args:
            question (`dict`): the input question
            candidates (`List[dict]`): the input candidates
            candidate_num (`int`): the number of candidates
            k (`int`): number of comparisons for each pair of candidates
        """
        candidates = candidates[:candidate_num]
        self.cache.load_knockout(
            instance_id=question["id"],
            n=candidate_num,
            k=k,
            category=question["category"],
        )
        round_num = 0
        knockout_traj = {
            "final": None,
            "detail": {},
        }
        while len(candidates) > 1:
            round_num += 1
            winners = []
            if len(candidates) % 2 == 1:
                winners.append(candidates[-1])
            pairs = []
            for i in range(1, len(candidates), 2):
                # pair-wise compare
                pairs.append(
                    self.judge.pairwise_compare(
                        question=question,
                        candidate_a=candidates[i - 1],
                        candidate_b=candidates[i],
                        k=k,
                    ),
                )
            rounds_detail = []
            for i, pair in tqdm(
                enumerate(pairs),
                desc=f"Round {round_num}",
                position=1,
                total=len(pairs),
            ):
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
            n=candidate_num,
            k=k,
            category=question["category"],
        )
        return candidates[0]

    def league(
        self,
        question: dict,
        candidates: List[dict],
        candidate_num: int,
        k: int,
    ) -> dict:
        """Run league competition."""
        # TBD
        pass
