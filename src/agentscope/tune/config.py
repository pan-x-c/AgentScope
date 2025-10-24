# -*- coding: utf-8 -*-
"""The configurations for agent learning."""

from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List
try:
    from omegaconf import OmegaConf
    from trinity.common.config import Config
except:
    pass

from .workflow import (
    WorkflowType,
)


class LearnProtocol(BaseModel):  # type: ignore [no-redef]
    # trainer selection
    trainer: str = Field(default="trinity")
    # agentflow name in trainer
    agentflow_name: str = Field(default="agentscope-evolver")
    # trainable targets defined in agentflow
    trainable_agent_targets: List[str] = Field(default=[])
    # Use dataset provided by the trainer (True: read each query from workflow input; False: AgentScope handles each query)
    external_dataset: bool = Field(default=True)
    # Use external environment provided by the trainer (True: read environment handle from input; False: AgentScope runs environment and tools)
    external_environment: bool = Field(default=False)
    # Use external reward provided by the trainer (True: compute reward outside AgentScope after workflow; False: AgentScope computes reward)
    external_reward: bool = Field(default=False)
    # trainer's config path
    trainer_config_path: str = Field(default="")

    async def agentscope_execute(self, task):
        raise NotImplementedError

# -*- coding: utf-8 -*-
"""The configurations for agent learning."""

from dataclasses import dataclass

from .workflow import (
    WorkflowType,
)


try:
    from omegaconf import OmegaConf
    from trinity.common.config import Config

    @dataclass
    class TuneConfig(Config):
        """Configuration for learning process."""

        def to_trinity_config(self, workflow_func: WorkflowType) -> Config:
            """Convert to Trinity-RFT compatible configuration."""
            workflow_name = "agentscope_workflow_adapter"
            self.buffer.explorer_input.taskset.default_workflow_type = (
                workflow_name
            )
            self.buffer.explorer_input.default_workflow_type = workflow_name
            self.buffer.explorer_input.taskset.workflow_args[
                "workflow_func"
            ] = workflow_func
            return self.check_and_update()

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

except ImportError:

    class TuneConfig:  # type: ignore [no-redef]
        """A placeholder class for TuneConfig when
        dependencies are missing."""

        def __init__(self) -> None:
            """Raise ImportError when instantiated."""
            raise ImportError(
                "Trinity-RFT or OmegaConf is not installed. "
                "Please install them with "
                "`pip install trinity-rft omegaconf`.",
            )

        @classmethod
        def load_config(cls, config_path: str) -> "TuneConfig":
            """Raise ImportError when trying to load configuration."""
            raise ImportError(
                "Trinity-RFT is not installed. Please "
                "install it with `pip install trinity-rft`.",
            )
