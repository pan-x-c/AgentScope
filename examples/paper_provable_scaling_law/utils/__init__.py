# -*- coding: utf-8 -*-
from .cache import Cache
from .dataset import Dataset, MMLUPro
from .worker import (
    Judge,
    MixedJudge,
    MMLUProJudge,
    MMLUProCoTJudge,
    Generator,
    MMLUProGenerator,
)

from .competition import Competition, Knockout, UCB


def get_generator(config: dict) -> Generator:
    """Get a generator from a config dict."""
    if config["type"] == "mmlu_pro":
        return MMLUProGenerator(config["model"])
    else:
        raise NotImplementedError


def get_judge(config: dict) -> Judge:
    """Get a judge from a config dict."""
    if config["type"] == "mmlu_pro_cot":
        return MMLUProCoTJudge(model_config=config["model"])
    if config["type"] == "mmlu_pro":
        return MMLUProJudge(model_config=config["model"])
    else:
        raise NotImplementedError


def get_dataset(config: dict) -> Dataset:
    """Get a dataset from a config dict."""
    if config["name"] == "mmlu_pro":
        return MMLUPro(
            max_instance=config["max_instance"],
            categories=config["categories"],
            split=config["split"],
        )
    else:
        raise NotImplementedError


def get_competition(
    config: dict,
    judge: MixedJudge,
    cache: Cache,
    to_dist: dict,
) -> Competition:
    """Create a competition from a config dict."""
    if config["method"] == "knockout":
        return Knockout(
            judge=judge,
            cache=cache,
            n=config["n"],
            k=config["k"],
            to_dist=to_dist,
        )
    if config["method"] == "ucb":
        return UCB(
            judge=judge,
            cache=cache,
            n=config["n"],
            k=config["k"],
            t=config["t"],
            to_dist=to_dist,
        )
    else:
        raise ValueError(f"Unknown competition method: {config['method']}")
