# -*- coding: utf-8 -*-
# preprocess and load datasets

from __future__ import annotations
import os
from typing import List
from abc import ABC, abstractmethod
from datasets import load_dataset
from tqdm import tqdm
import jsonlines


DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")


class Dataset(ABC):
    @classmethod
    @abstractmethod
    def preprocess(cls) -> None:
        pass

    @classmethod
    @abstractmethod
    def format_sample(cls, sample: dict) -> dict:
        """Format a sample into a dict."""
        pass

    @classmethod
    def from_dict(cls, config: dict) -> Dataset:
        if config["name"] == "mmlu_pro":
            return MMLUPro(
                max_instance=config["max_instance"],
                categories=config["categories"],
            )
        else:
            raise NotImplementedError(f"Dataset {config['name']} not found.")

    @abstractmethod
    def calculate_stats(self, sample: dict, candidates: List[dict]) -> dict:
        pass


class MMLUPro(Dataset):
    PROMPT_TEMPLATE = """
Question: {question}
Options:
{options}
"""

    OPTION_TEMPLATE = """({option}) {context}"""

    def __init__(self, categories: List[str], max_instance: int):
        self.categories = categories
        self.max_instance = max_instance

    @classmethod
    def preprocess(cls) -> None:
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        categories = set(ds["test"].unique("category"))
        os.makedirs(
            os.path.join(DATASET_DIR, "mmlu_pro", "test"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(DATASET_DIR, "mmlu_pro", "validation"),
            exist_ok=True,
        )
        for category in categories:
            test_filtered = ds["test"].filter(
                lambda example: example["category"] == category,
            )
            vali_filtered = ds["validation"].filter(
                lambda example: example["category"] == category,
            )
            category = category.replace(" ", "_").lower()
            test_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "mmlu_pro",
                    "test",
                    f"{category}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            vali_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "mmlu_pro",
                    "validation",
                    f"{category}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            print(f"Saved test and validation data for category: {category}")

    @classmethod
    def format_sample(cls, sample: dict) -> dict:
        options = []
        for i, option in enumerate(sample["options"]):
            options.append(
                cls.OPTION_TEMPLATE.format(option=chr(65 + i), context=option),
            )
        return {
            "id": sample["question_id"],
            "category": sample["category"],
            "question": cls.PROMPT_TEMPLATE.format(
                question=sample["question"],
                options="\n".join(options),
            ),
            "answer": sample["answer"],
        }

    def calculate_stats(self, sample: dict, candidates: List[dict]) -> dict:
        def _get_majority(candidates: list[dict]) -> str:
            votes = {}
            for c in candidates:
                if c["answer"] not in votes:
                    votes[c["answer"]] = 0
                else:
                    votes[c["answer"]] += 1
            return max(votes, key=votes.get)

        stats = {
            "target": sample["answer"],
            "question": sample["question"],
        }
        total = 0
        correct = 0
        for candidate in candidates:
            total += 1
            if candidate["answer"] == sample["answer"]:
                correct += 1
        stats["acc"] = correct / total
        stats["total_cnt"] = total
        stats["correct_cnt"] = correct
        stats["majority"] = _get_majority(candidates)
        return stats

    def _load_category(self, category: str, instance_num: int) -> List[dict]:
        file_path = os.path.join(
            DATASET_DIR,
            "mmlu_pro",
            "test",
            f"{category}.jsonl",
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for category: {category}")
        instances = []
        with jsonlines.open(file_path) as reader:
            for i, line in enumerate(reader):
                if i >= instance_num:
                    break
                instances.append(line)
        return instances

    def __iter__(self) -> Dataset:
        self.cur_category_index = 0
        self.cur_instance_index = 0
        self.total_samples = sum(
            len(self._load_category(category, self.max_instance))
            for category in self.categories
        )
        self.samples = self._load_category(
            self.categories[0],
            self.max_instance,
        )
        self.pbar = tqdm(total=self.total_samples, desc="Dataset", position=0)
        return self

    def __next__(self) -> dict:
        if self.cur_instance_index >= len(self.samples):
            self.cur_instance_index = 0
            self.cur_category_index += 1
            if self.cur_category_index >= len(self.categories):
                self.pbar.close()
                raise StopIteration
            self.samples = self._load_category(
                self.categories[self.cur_category_index],
                self.max_instance,
            )
        sample = self.samples[self.cur_instance_index]
        self.cur_instance_index += 1
        self.pbar.update(1)
        return self.format_sample(sample)


if __name__ == "__main__":
    samples = MMLUPro.load(category="physics", max_instance=1)
    print(samples[0])
    print(MMLUPro.format_sample(samples[0])["question"])
