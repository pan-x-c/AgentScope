# -*- coding: utf-8 -*-
"""Dataset definition for tuner."""
from typing import Optional, Dict
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Dataset information for tuning.
    Compatible with huggingface dataset format.
    Agentscope will load the dataset from the given path using
    `datasets.load_dataset`.
    """

    path: str = Field(
        description="Path to your dataset.",
    )
    name: Optional[str] = Field(
        description="The name of the dataset configuration.",
        default=None,
    )
    split: Optional[str] = Field(
        description="The dataset split to use.",
        default="train",
    )

    def preview(self, n: int = 5) -> Dict:
        """Preview the dataset information.

        Args:
            n (int): Number of samples to preview.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "The `datasets` library is not installed. "
                "Please install it with `pip install datasets`.",
            ) from e
        import json

        ds = load_dataset(
            path=self.path,
            name=self.name,
            split=self.split,
        )
        samples = ds[:n]
        print(json.dumps(samples, indent=2))
        return samples
