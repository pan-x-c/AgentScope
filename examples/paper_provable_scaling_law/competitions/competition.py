# -*- coding: utf-8 -*-
"""Competition module."""
# pylint: disable=E0611,C0411

from __future__ import annotations
from abc import abstractmethod
from typing import List, Any
from loguru import logger
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

    def calculate_stats(
        self,
        dataset: Dataset,
    ) -> None:
        """Calculate the stats of the competition.

        Args:
            cache (`Cache`): the cache object
            dataset (`Dataset`): the dataset object
        """
        logger.info("Calculating competition stats...")
        competition_stats = self.default_competition_stats()

        for question in dataset:
            question_stats = self.get_question_stats(question)
            self.update_competition_stats(
                competition_stats=competition_stats,
                category=question["category"],
                question_stats=question_stats,
            )

        self.save_competition_stats(competition_stats)

        logger.info("Finished calculating competition stats")

    @abstractmethod
    def default_competition_stats(
        self,
    ) -> dict:
        """Get the default stats of the competition."""

    @abstractmethod
    def get_question_stats(
        self,
        question: dict,
    ) -> dict:
        """Get the stats of a question."""

    @abstractmethod
    def update_competition_stats(
        self,
        competition_stats: dict,
        category: str,
        question_stats: dict,
    ) -> None:
        """Update the stats of the competition."""

    @abstractmethod
    def save_competition_stats(
        self,
        competition_stats: dict,
    ) -> None:
        """Save the stats of the competition."""
