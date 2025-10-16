# -*- coding: utf-8 -*-
"""The configurations for agent learning."""

from typing import TYPE_CHECKING

from dataclasses import dataclass

from .workflow import WorkflowType, TRAINABLE_WORKFLOW_NAME

if TYPE_CHECKING:
    from omegaconf import OmegaConfig
    from trinity.common.config import Config
else:
    Config = "trinity.common.config.Config"


@dataclass
class LearnConfig(Config):
    """Configuration for learning process."""

    def to_trinity_config(self, workflow_func: WorkflowType) -> Config:
        """Convert to Trinity-RFT compatible configuration."""
        self.buffer.explorer_input.taskset.default_workflow_type = (
            TRAINABLE_WORKFLOW_NAME
        )
        self.buffer.explorer_input.taskset.workflow_args = {
            "workflow_func": workflow_func,
        }
        return self.check_and_update()

    @classmethod
    def load_config(cls, config_path: str) -> "LearnConfig":
        """Load the learning configuration from a YAML file.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            LearnConfig: The loaded learning configuration.
        """
        schema = OmegaConfig.structured(cls)
        yaml_config = OmegaConfig.load(config_path)
        try:
            config = OmegaConfig.merge(schema, yaml_config)
            return OmegaConfig.to_object(config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e
