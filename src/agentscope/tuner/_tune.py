# -*- coding: utf-8 -*-
"""The main entry point for agent learning."""
from ._workflow import WorkflowType
from ._judge import JudgeType
from ._model import TunerChatModel
from ._dataset import Dataset
from ._config import (
    to_trinity_config,
    check_judge_function,
    check_workflow_function,
)
from ._algorithm import Algorithm


def tune(
    *,
    workflow_func: WorkflowType,
    judge_func: JudgeType | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    model: TunerChatModel | None = None,
    auxiliary_models: dict[str, TunerChatModel] | None = None,
    algorithm: Algorithm | None = None,
    config_path: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Train the agent workflow with the specific configuration.

    Args:
        workflow_func (WorkflowType): The learning workflow function
            to execute.
        judge_func (JudgeType, optional): The judge function used to
            evaluate the workflow output. Defaults to None.
        train_dataset (Dataset, optional): The training dataset for
            the learning process. Defaults to None.
        eval_dataset (Dataset, optional): The evaluation dataset for
            the learning process. Defaults to None.
        model (TunerChatModel, optional): The chat model to be tuned.
            Defaults to None.
        auxiliary_models (dict[str, TunerChatModel], optional): A
            dictionary of auxiliary chat models for LLM-as-a-Judge
            or acting other agents in multi-agent scenarios.
            Defaults to None.
        algorithm (Algorithm, optional): The tuning algorithm
            configuration. Defaults to None.
        config_path (str, optional): Path to the learning configuration
            file. Defaults to None.
    """
    try:
        from trinity.cli.launcher import run_stage
    except ImportError as e:
        raise ImportError(
            "Trinity-RFT is not installed. Please install it with "
            "`pip install trinity-rft`.",
        ) from e

    check_workflow_function(workflow_func)
    if judge_func is not None:
        check_judge_function(judge_func)

    config = to_trinity_config(
        config_path=config_path,
        workflow_func=workflow_func,
        judge_func=judge_func,
        model=model,
        auxiliary_models=auxiliary_models,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        algorithm=algorithm,
        experiment_name=experiment_name,
    )

    return run_stage(
        config=config,
    )
