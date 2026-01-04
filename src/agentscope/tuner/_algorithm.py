# -*- coding: utf-8 -*-
"""Algorithm definition for tuner."""

from pydantic import BaseModel, Field


class Algorithm(BaseModel):
    """Algorithm information for tuning."""

    algorithm_type: str = Field(
        description=("The tuning algorithm type " "e.g., 'grpo', 'sft'"),
        default="multi_step_grpo",
    )
    learning_rate: float = Field(
        description="The learning rate for the algorithm.",
        default=1e-6,
    )
    group_size: int = Field(
        description=(
            "The group size for algorithms "
            "required group rollout, e.g., GRPO."
        ),
        default=8,
    )
    batch_size: int = Field(
        description="The batch size for the algorithm.",
        default=32,
    )
