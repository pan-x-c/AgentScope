# -*- coding: utf-8 -*-
"""Configuration conversion for tuner."""
from typing import Any
from pathlib import Path
from datetime import datetime

from ._workflow import WorkflowType
from ._judge import JudgeType
from ._model import TunerChatModel
from ._dataset import Dataset
from ._algorithm import Algorithm


def to_trinity_config(
    *,
    config_path: str | None = None,
    workflow_func: WorkflowType | None = None,
    judge_func: JudgeType | None = None,
    model: TunerChatModel | None = None,
    auxiliary_models: dict[str, TunerChatModel] | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    algorithm: Algorithm | None = None,
    experiment_name: str | None = None,
) -> Any:
    """Convert to Trinity-RFT compatible configuration."""
    from trinity.common.config import (
        Config,
        TasksetConfig,
        load_config,
        InferenceModelConfig,
    )

    auto_config = False
    if config_path is None:
        temp_path = Path(__file__).parent / "template" / "config.yaml"
        config_path = str(temp_path.absolute())
        auto_config = True

    config = load_config(config_path)
    assert isinstance(config, Config), "Loaded config is not valid."

    if experiment_name is None and auto_config:
        experiment_name = "Experiment-" + datetime.now().strftime(
            "%Y%m%d%H%M%S",
        )
        config.name = experiment_name

    workflow_name = "agentscope_workflow_adapter_v1"
    if train_dataset is not None:
        config.buffer.explorer_input.taskset = TasksetConfig(
            name="train_taskset",
            path=train_dataset.path,
            split=train_dataset.split,
            subset_name=train_dataset.name,
        )
        config.buffer.total_epochs = train_dataset.total_epochs
        config.buffer.total_steps = train_dataset.total_steps
    config.buffer.explorer_input.taskset.default_workflow_type = workflow_name
    config.buffer.explorer_input.default_workflow_type = workflow_name
    config.buffer.explorer_input.taskset.workflow_args[
        "workflow_func"
    ] = workflow_func

    if model is not None:
        model_config = model.get_config()
        config.model.model_path = model_config["model_path"]
        config.model.max_model_len = model_config["max_model_len"]
        config.model.max_response_tokens = model.max_tokens
        config.explorer.rollout_model = InferenceModelConfig(
            **model.get_config(),
        )
        config.explorer.rollout_model.enable_history = True
    if auxiliary_models is not None:
        for name, aux_chat_model in auxiliary_models.items():
            model_config = InferenceModelConfig(
                **aux_chat_model.get_config(),
            )
            model_config.name = name
            config.explorer.auxiliary_models.append(
                model_config,
            )
    if judge_func is not None:
        config.buffer.explorer_input.taskset.workflow_args[
            "judge_func"
        ] = judge_func
    if eval_dataset is not None:
        config.buffer.explorer_input.eval_tasksets.append(
            TasksetConfig(
                name="eval_taskset",
                path=eval_dataset.path,
                split=eval_dataset.split,
                subset_name=eval_dataset.name,
            ),
        )
    if algorithm is not None:
        config.algorithm.algorithm_type = algorithm.algorithm_type
        config.algorithm.repeat_times = algorithm.group_size
        config.algorithm.optimizer.lr = algorithm.learning_rate
        config.buffer.batch_size = algorithm.batch_size
    return config.check_and_update()
