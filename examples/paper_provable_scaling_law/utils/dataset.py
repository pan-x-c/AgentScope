# -*- coding: utf-8 -*-
"""preprocess and load datasets"""

from __future__ import annotations
import os
import re
import random
from typing import List, Tuple
from abc import ABC, abstractmethod
from datasets import load_dataset
from tqdm import tqdm
import jsonlines


DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")


class Dataset(ABC):
    """Base class for datasets."""

    @classmethod
    @abstractmethod
    def preprocess(cls) -> None:
        """Preprocess the dataset."""

    @classmethod
    @abstractmethod
    def format_sample(cls, sample: dict) -> dict:
        """Format a sample into a dict."""

    @abstractmethod
    def calculate_stats(self, sample: dict, candidates: List[dict]) -> dict:
        """Calculate statistics for a sample and its candidates."""


class MMLUPro(Dataset):
    """MMLU-Pro dataset."""

    PROMPT_TEMPLATE = """
Question: {question}
Options:
{options}
"""

    OPTION_TEMPLATE = """({option}) {context}"""

    def __init__(
        self,
        categories: List[str],
        max_instance: int,
        split: str = "random",
    ):
        self.categories = categories
        self.max_instance = max_instance
        self.split = split
        self.cur_category_index = 0
        self.cur_instance_index = 0
        self.total_samples = 0
        self.samples = []
        self.pbar = None

    def __reduce__(self) -> Tuple:
        return (
            self.__class__,
            (self.categories, self.max_instance, self.split),
        )

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
                lambda example: example["category"]
                == category,  # pylint: disable=cell-var-from-loop
            )
            vali_filtered = ds["validation"].filter(
                lambda example: example["category"]
                == category,  # pylint: disable=cell-var-from-loop
            )
            ct = category.replace(" ", "_").lower()
            test_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "mmlu_pro",
                    "test",
                    f"{ct}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            vali_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "mmlu_pro",
                    "validation",
                    f"{ct}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            print(f"Saved test and validation data for category: {ct}")

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
            return max(votes, key=votes.get)  # type: ignore[arg-type]

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
            self.split,
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


class GPQA(Dataset):
    """
    Modified from:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/cot_zeroshot/utils.py
    """

    def __init__(
        self,
        categories: List[str],
        max_instance: int,
        split: str = "random",
    ):
        self.categories = categories
        self.max_instance = max_instance
        self.split = split
        self.cur_category_index = 0
        self.cur_instance_index = 0
        self.total_samples = 0
        self.samples = []
        self.pbar = None

    def __reduce__(self) -> Tuple:
        return (
            self.__class__,
            (self.categories, self.max_instance, self.split),
        )

    @classmethod
    def preprocess(cls) -> None:
        def _preprocess_text(text: str) -> str:
            if text is None:
                return " "
            text = text.strip()
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\]", "", text)
            text = text.replace("  ", " ")
            return text

        def _process_doc(
            doc: dict,
            idx: int,
            category: str,
        ) -> dict:
            choices = [
                _preprocess_text(doc["Incorrect Answer 1"]),
                _preprocess_text(doc["Incorrect Answer 2"]),
                _preprocess_text(doc["Incorrect Answer 3"]),
                _preprocess_text(doc["Correct Answer"]),
            ]

            random.shuffle(choices)
            correct_answer_index = choices.index(
                _preprocess_text(doc["Correct Answer"]),
            )

            out_doc = {
                "id": idx,
                "question": _preprocess_text(doc["Question"]),
                "choice1": choices[0],
                "choice2": choices[1],
                "choice3": choices[2],
                "choice4": choices[3],
                "answer": f"{chr(65 + correct_answer_index)}",
                "category": category,
            }
            return out_doc

        from functools import partial

        gpqa_diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond")[
            "train"
        ].map(partial(_process_doc, category="diamond"), with_indices=True)
        gpqa_main = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"].map(
            partial(_process_doc, category="main"),
            with_indices=True,
        )
        gpqa_extended = load_dataset("Idavidrein/gpqa", "gpqa_extended")[
            "train"
        ].map(partial(_process_doc, category="extended"), with_indices=True)
        os.makedirs(os.path.join(DATASET_DIR, "gpqa", "train"), exist_ok=True)
        gpqa_diamond.to_json(
            os.path.join(DATASET_DIR, "gpqa", "train", "diamond.json"),
        )
        gpqa_main.to_json(
            os.path.join(DATASET_DIR, "gpqa", "train", "main.json"),
        )
        gpqa_extended.to_json(
            os.path.join(DATASET_DIR, "gpqa", "train", "extended.json"),
        )

    PROMPT_TEMPLATE = """
Question:
{question}

Choices:
{choices}
"""

    OPTION_TEMPLATE = """({choice}) {content}"""

    @classmethod
    def format_sample(cls, sample: dict) -> dict:
        choices = [
            cls.OPTION_TEMPLATE.format(
                choice=chr(65 + i),
                content=sample["choice" + str(i + 1)],
            )
            for i in range(4)
        ]
        return {
            "id": sample["id"],
            "category": sample["category"],
            "question": cls.PROMPT_TEMPLATE.format(
                question=sample["question"],
                choices="\n".join(choices),
            ),
            "answer": sample["answer"],
        }

    def _load_category(self, category: str, instance_num: int) -> List[dict]:
        file_path = os.path.join(
            DATASET_DIR,
            "gpqa",
            self.split,
            f"{category}.json",
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

    @classmethod
    def calculate_stats(cls, sample: dict, candidates: List[dict]) -> dict:
        def _get_majority(candidates: list[dict]) -> str:
            votes = {}
            for c in candidates:
                if c["answer"] not in votes:
                    votes[c["answer"]] = 0
                else:
                    votes[c["answer"]] += 1
            return max(votes, key=votes.get)  # type: ignore[arg-type]

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
