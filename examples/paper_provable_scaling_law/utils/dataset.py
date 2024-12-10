# -*- coding: utf-8 -*-
"""preprocess and load datasets"""

from __future__ import annotations
import os
import re
from typing import List, Optional
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

    def __init__(self, categories: List[str], max_instance: int, split: str = "random"):
        self.categories = categories
        self.max_instance = max_instance
        self.split = split
        self.cur_category_index = 0
        self.cur_instance_index = 0
        self.total_samples = 0
        self.samples = []
        self.pbar = None

    def __reduce__(self):
        return (self.__class__, (self.categories, self.max_instance, self.split))

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


class MATH(Dataset):
    """MATH dataset."""

    PROMPT_TEMPLATE = """
    """

    @classmethod
    def preprocess(cls) -> None:
        ds = cls.process_docs(load_dataset("lighteval/MATH-Hard"))
        categories = set(ds["test"].unique("category"))
        os.makedirs(
            os.path.join(DATASET_DIR, "MATH_lv5", "test"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(DATASET_DIR, "MATH_lv5", "train"),
            exist_ok=True,
        )
        for category in categories:
            test_filtered = ds["test"].filter(
                lambda example: example["type"] == category,
            )
            vali_filtered = ds["train"].filter(
                lambda example: example["type"] == category,
            )
            category = category.replace(" ", "_")
            test_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "MATH_lv5",
                    "test",
                    f"{category}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            vali_filtered.to_json(
                os.path.join(
                    DATASET_DIR,
                    "MATH_lv5",
                    "train",
                    f"{category}.jsonl",
                ),
                lines=True,
                force_ascii=False,
            )
            print(f"Saved test and train data for category: {category}")

    @classmethod
    def process_docs(cls, dataset: Dataset) -> Dataset:
        def _process_doc(doc: dict, idx: int) -> dict:
            out_doc = {
                "id": idx,
                "question": doc["problem"],
                "solution": doc["solution"],
                "answer": cls.normalize_final_answer(
                    cls.remove_boxed(
                        cls.last_boxed_only_string(doc["solution"]),
                    ),
                ),
                "category": doc["type"].replace(" ", "_"),
            }
            if getattr(doc, "few_shot", None) is not None:
                out_doc["few_shot"] = True
            return out_doc

        return dataset.map(_process_doc, with_indices=True)

    INVALID_ANSWER = "[invalidanswer]"

    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
    ]

    @classmethod
    def normalize_final_answer(cls, final_answer: str) -> str:
        """
        Normalize a final answer to a quantitative reasoning question.

        Copied character for character from appendix D of Lewkowycz et al. (2022)
        """
        final_answer = final_answer.split("=")[-1]

        for before, after in MATH.SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in MATH.REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX:
        #  \fracab -> \frac{a}{b}
        #  \frac{abc}{bef} -> \frac{abc}{bef}
        #  \fracabc -> \frac{a}{b}c
        #  \sqrta -> \sqrt{a}
        #  \sqrtab -> sqrt{a}b
        final_answer = re.sub(
            r"(frac)([^{])(.)",
            "frac{\\2}{\\3}",
            final_answer,
        )
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    @classmethod
    def last_boxed_only_string(cls, string: str) -> Optional[str]:
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    @classmethod
    def remove_boxed(cls, s: str) -> str:
        try:
            if "\\boxed " in s:
                left = "\\boxed "
                assert s[: len(left)] == left
                return s[len(left) :]

            left = "\\boxed{"

            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        except AssertionError:
            return MATH.INVALID_ANSWER


if __name__ == "__main__":
    MATH.preprocess()
