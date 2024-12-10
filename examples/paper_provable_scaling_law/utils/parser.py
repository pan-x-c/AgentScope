# -*- coding: utf-8 -*-
"""Parser to parse values from the response of LLM."""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import List, Any
from loguru import logger


class Parser(ABC):
    """A base parser to parse the LLM response"""

    @abstractmethod
    def parse_to_dict(self, raw: str) -> dict:
        """Parse the raw response as a dict"""

    @property
    @abstractmethod
    def format_instruction(self) -> str:
        """
        The format instruction which can be attached to the end of the prompt
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, config: dict) -> Parser:
        """Load the parser from dict config"""


class TagParser(Parser):
    """A parser that extracts content inside <{tag_name}></{tag_name}>"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.begin = f"<{name}>"
        self.end = f"</{name}>"
        self.description = description
        self._instruction = f"{self.begin}\n{description}\n{self.end}"

    def extract(self, raw: str) -> str:
        left = raw.find(self.begin)
        if left == -1:
            logger.error(f"Tag {self.begin} not found in the content:\n{raw}")
            raise ValueError(
                f"Tag {self.begin} not found in the content:\n{raw}",
            )
        right = raw.find(self.end)
        if right == -1:
            logger.error(
                f"Tag {self.end} end not found in the content:\n{raw}",
            )
            raise ValueError(
                f"Tag {self.end} not found in the content:\n{raw}",
            )
        return raw[left + len(self.begin) : right].strip()

    def parse(self, text: str) -> Any:
        return text

    def parse_to_dict(self, raw: str) -> dict:
        return {self.name: self.parse(self.extract(raw))}

    def construct(self, res: dict) -> str:
        return f"{self.begin}\n{res[self.name]}\n{self.end}"

    @property
    def format_instruction(self) -> str:
        return self._instruction

    @classmethod
    def from_dict(cls, config: dict) -> TagParser:
        return TagParser(config["name"], config["description"])


class MMLUProParser(TagParser):
    """
    Modified from MMLU-Pro
    https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/compute_accuracy.py
    """

    def __init__(
        self,
        name: str,
    ):
        super().__init__(
            name,
            '"the answer is (X)" where X is the correct letter choice',
        )

    def extract_again(self, text: str) -> str:
        match = re.search(r".*[aA]nswer:\s*([A-J])", text)
        if match:
            return match.group(1)
        else:
            return self.extract_final(text)

    def extract_final(self, text: str) -> str:
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return "Uncertain"

    def parse(self, text: str) -> Any:
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return self.extract_again(text)


class ChoiceParser(TagParser):
    """
    Parse result from specific choices.
    """

    def __init__(
        self,
        name: str,
        choices: List[str],
    ):
        super().__init__(
            name,
            " or ".join(choices),
        )
        self.choices = choices

    def parse(self, text: str) -> Any:
        if text in self.choices:
            return text
        else:
            return "Uncertain"


class PairWiseParser(ChoiceParser):
    """
    Pair-wise comparison parser used in pairwise comparison task.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(
            "winner",
            choices=[
                "Solution 1",
                "Solution 2",
                "Tie",
            ],
        )

    def parse(self, text: str) -> Any:
        if text == "Solution 1":
            score = [1, 0]
        elif text == "Solution 2":
            score = [0, 1]
        else:
            score = [0.5, 0.5]
        return {
            "a": score[0],
            "b": score[1],
        }


class MultiTagsParser(Parser):
    """A parser that can parse multiple tags."""

    def __init__(self, tags: List[TagParser]) -> None:
        self.tags = tags
        self._instruction = "\n".join(tag.format_instruction for tag in tags)

    def construct(self, res: dict) -> str:
        return "\n".join([tag.construct(res) for tag in self.tags])

    def parse_to_dict(self, raw: str) -> dict:
        result = {}
        for i in range(len(self.tags)):
            begin = self.tags[i].begin
            end = self.tags[i].end
            left = raw.find(begin)
            right = raw.find(end)
            if left == -1:
                logger.error(f"Tag {begin} not found in the content:\n{raw}")
                text = ""
            elif right == -1:
                if i < len(self.tags) - 1:
                    logger.warning(
                        f"Tag {end} end not found, "
                        f"try to find {self.tags[i + 1].begin} instead.",
                    )
                    right = raw.find(self.tags[i + 1].begin)
                    if right == -1:
                        logger.error(
                            f"Tag {self.tags[i + 1].begin} not found in the content:\n{raw}",
                        )
                        text = ""
                    else:
                        text = raw[left + len(begin) : right].strip()
                elif i == len(self.tags) - 1:
                    text = raw[left + len(begin) :].strip()
                else:
                    logger.error(
                        f"Tag {end} end not found in the content:\n{raw}",
                    )
                    text = ""
            else:
                text = raw[left + len(begin) : right].strip()
            result[self.tags[i].name] = self.tags[i].parse(text)
        return result

    @property
    def format_instruction(self) -> str:
        return self._instruction

    @classmethod
    def from_dict(cls, config: dict) -> MultiTagsParser:
        tags = [TagParser.from_dict(tag) for tag in config["tags"]]
        return MultiTagsParser(tags)
