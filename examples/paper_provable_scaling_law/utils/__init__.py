# -*- coding: utf-8 -*-
from .cache import Cache
from .dataset import Dataset, MMLUPro, GPQA
from .worker import (
    Judge,
    MixedJudge,
    MMLUProJudge,
    MMLUProCoTJudge,
    Generator,
    MMLUProGenerator,
    GPQAGenerator,
)


def get_generator(config: dict) -> Generator:
    """Get a generator from a config dict."""
    if config["type"] == "mmlu_pro":
        return MMLUProGenerator(config["model"])
    elif config["type"] == "gpqa":
        return GPQAGenerator(config["model"])
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
    if config["name"] == "gpqa":
        return GPQA(
            max_instance=config["max_instance"],
            categories=config["categories"],
            split=config["split"],
        )
    else:
        raise NotImplementedError
