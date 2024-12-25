# -*- coding: utf-8 -*-
"""Competition module."""
# pylint: disable=E0611,C0411

from __future__ import annotations
from abc import abstractmethod
from typing import List, Any
from agentscope.rpc import async_func, RpcMeta

from utils.worker import MixedJudge
from utils.cache import Cache
from utils.dataset import Dataset


class Competition(metaclass=RpcMeta):
    """A collection of competitions."""

    _COMPETITIONS = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Register a competition."""

        def decorator(subcls: type) -> type:
            cls._COMPETITIONS[name] = subcls
            return subcls

        return decorator

    @classmethod
    def create(
        cls,
        judge: MixedJudge,
        cache: Cache,
        config: dict,
    ) -> Competition:
        """Create a competition instance."""
        name = config.pop("method", "")
        subcls = cls._COMPETITIONS.get(name)
        if not subcls:
            raise ValueError(f"Unknown competition type: {name}")
        return subcls(judge=judge, cache=cache, **config)

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
