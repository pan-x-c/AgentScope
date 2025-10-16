# -*- coding: utf-8 -*-
"""The main entry point for agent learning."""
from typing import TYPE_CHECKING

from .workflow import WorkflowType
from .config import LearnConfig


if TYPE_CHECKING:
    from trinity.cli.launcher import run_stage
else:
    run_stage = None  # type: ignore


def learn(workflow_type: WorkflowType, config: LearnConfig) -> None:
    """Train the agent workflow with the specific configuration.

    Args:
        workflow (WorkflowType): The learning workflow class or function
            to execute.
        config (LearnConfig): The configuration for the learning process.
    """
    return run_stage(
        config.to_trinity_config(workflow_func=workflow_type),
        config.ray_address,
    )
