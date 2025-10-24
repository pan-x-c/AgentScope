# -*- coding: utf-8 -*-
"""The main entry point for agent learning."""
from .workflow import (
    WorkflowType,
    validate_function_signature,
)
from .config import TuneConfig

def validate_dependency(config: LearnConfig) -> None:
    if config.trainer == "trinity":
        try:
            import trinity  # noqa: F401
            from trinity.cli.launcher import run_stage
        except ImportError as e:
            raise ImportError(
                "Trinity-RFT is not installed. Please install it with "
                "`pip install trinity-rft`.",
            ) from e
    if config.trainer == "trinity-agentevolver":
        try:
            import trinity_agentevolver  # noqa: F401
            from trinity_agentevolver.cli.launcher import run_stage
        except ImportError as e:
            raise ImportError(
                "Trinity-AgentEvolver is not installed. Please install it with "
                "`pip install trinity-agentevolver`.",
            ) from e
    else:
        raise NotImplementedError(
            f"Trainer {config.trainer} is not supported.",
        )

def get_trainer(config: LearnConfig) -> None:
    validate_dependency(config)
    if config.trainer == "trinity":
        from trinity.cli.launcher import run_stage
        def trainer_execute(workflow_func):
            return run_stage(
                config.to_trinity_config(workflow_func=workflow_func),
            )
        return trainer_execute
    if config.trainer == "trinity-agentevolver":
        from trinity_agentevolver.cli.launcher import run_stage
        def trainer_execute(workflow_func):
            return run_stage(
                config.to_trinity_agentevolver_config(
                    workflow_func=workflow_func,
                ),
            )
        return trainer_execute

def tune(workflow_func: WorkflowType, config: TuneConfig) -> None:
    """Train the agent workflow with the specific configuration.

    Args:
        workflow_func (WorkflowType): The learning workflow function
            to execute.
        config (TuneConfig): The configuration for the learning process.
    """

    if not validate_function_signature(workflow_func):
        raise ValueError(
            "Invalid workflow function signature, please "
            "check the types of your workflow input/output.",
        )

    trainer_execute = get_trainer(config)

    trainer_execute(workflow_func)
