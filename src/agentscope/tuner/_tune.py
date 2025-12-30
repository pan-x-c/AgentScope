# -*- coding: utf-8 -*-
"""The main entry point for agent learning."""
from dataclasses import dataclass
from typing import Callable, Any
from ._types import (
    WorkflowType,
    JudgeType,
    TunerChatModel,
    Dataset,
)


def tune(
    *,
    workflow_func: Callable,
    model: TunerChatModel,
    train_dataset: Dataset,
    config_path: str,
    auxiliary_models: dict[str, TunerChatModel] | None = None,
    judge_func: Callable | None = None,
    eval_dataset: Dataset | None = None,
) -> None:
    """Train the agent workflow with the specific configuration.

    Args:
        workflow_func (WorkflowType): The learning workflow function
            to execute.
        train_dataset (Dataset): The training dataset.
        config_path (str): The configuration for the learning process.
        judge_func (JudgeType | None): The judge function to evaluate
            agent responses. (Optional)
        eval_dataset (Dataset | None): The evaluation dataset. (Optional)
    """
    try:
        from trinity.cli.launcher import run_stage
        from trinity.common.config import Config
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            "Trinity-RFT is not installed. Please install it with "
            "`pip install trinity-rft`.",
        ) from e

    @dataclass
    class TuneConfig(Config):
        """Configuration for learning process."""

        def to_trinity_config(
            self,
            workflow_func: WorkflowType,
            train_dataset: Dataset | None,
            model: TunerChatModel | None,
            judge_func: JudgeType | None,
            eval_dataset: Dataset | None,
            auxiliary_models: dict[str, TunerChatModel] | None = None,
        ) -> Config:
            """Convert to Trinity-RFT compatible configuration."""
            from trinity.common.config import TasksetConfig

            workflow_name = "agentscope_workflow_adapter_v1"
            if train_dataset is not None:
                self.buffer.explorer_input.taskset = TasksetConfig(
                    name="train_taskset",
                    path=train_dataset.path,
                    split=train_dataset.split,
                    subset_name=train_dataset.name,
                )
            self.buffer.explorer_input.taskset.default_workflow_type = (
                workflow_name
            )
            self.buffer.explorer_input.default_workflow_type = workflow_name
            self.buffer.explorer_input.taskset.workflow_args[
                "workflow_func"
            ] = workflow_func

            if model is not None:
                model_config = model.get_config()
                self.model.model_path = model_config["model_path"]
                self.model.max_model_len = model_config["max_model_len"]
                self.model.max_response_tokens = model.max_tokens
                self.explorer.rollout_model = self.get_model_config(model)
                self.explorer.rollout_model.enable_history = True
            if auxiliary_models is not None:
                for name, aux_chat_model in auxiliary_models.items():
                    model_config = self.get_model_config(aux_chat_model)
                    model.model_name = name
                    self.explorer.auxiliary_models.append(
                        model_config,
                    )
            if judge_func is not None:
                self.buffer.explorer_input.taskset.workflow_args[
                    "judge_func"
                ] = judge_func
            if eval_dataset is not None:
                self.buffer.explorer_input.eval_tasksets.append(
                    TasksetConfig(
                        name="eval_taskset",
                        path=eval_dataset.path,
                        split=eval_dataset.split,
                        subset_name=eval_dataset.name,
                    ),
                )
            return self.check_and_update()

        def get_model_config(
            self,
            chat_model: TunerChatModel,
        ) -> Any:
            """Get the model configuration for Trinity-RFT."""
            from trinity.common.config import InferenceModelConfig

            return InferenceModelConfig(
                **chat_model.get_config(),
            )

        @classmethod
        def load_config(cls, config_path: str) -> "TuneConfig":
            """Load the learning configuration from a YAML file.

            Args:
                config_path (str): The path to the configuration file.

            Returns:
                TuneConfig: The loaded learning configuration.
            """
            schema = OmegaConf.structured(cls)
            yaml_config = OmegaConf.load(config_path)
            try:
                config = OmegaConf.merge(schema, yaml_config)
                return OmegaConf.to_object(config)
            except Exception as e:
                raise ValueError(f"Invalid configuration: {e}") from e

    return run_stage(
        config=TuneConfig.load_config(config_path).to_trinity_config(
            workflow_func=workflow_func,
            train_dataset=train_dataset,
            model=model,
            judge_func=judge_func,
            eval_dataset=eval_dataset,
            auxiliary_models=auxiliary_models,
        ),
    )
