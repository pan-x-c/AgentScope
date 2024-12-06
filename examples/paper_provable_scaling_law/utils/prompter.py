# -*- coding: utf-8 -*-
from typing import List
from .parser import Parser


class GenerationPrompter:
    """A class for generating prompts for generation tasks."""

    def __init__(
        self,
        prompt: str,
        parser: Parser,
        sys_prompt: str = None,
    ) -> None:
        self.prompt = prompt
        self.parser = parser
        self.sys_prompt = sys_prompt

    def generate_prompt(self, question: str) -> List:
        if self.sys_prompt is None:
            return [
                {
                    "name": "user",
                    "role": "user",
                    "content": self.prompt.format(
                        question=question,
                        format=self.parser.format_instruction,
                    ),
                },
            ]
        else:
            return [
                {
                    "name": "system",
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "name": "user",
                    "role": "user",
                    "content": self.prompt.format(
                        question=question,
                        format=self.parser.format_instruction,
                    ),
                },
            ]

    def parse_text(self, text: str) -> dict:
        result = {}
        try:
            result = self.parser.parse_to_dict(text)
        except Exception:
            pass
        result["raw"] = text
        return result


class ComparisonPrompter:
    """A class for generating prompts for comparison tasks."""

    def __init__(
        self,
        prompt: str,
        parser: Parser,
        sys_prompt: str = None,
    ) -> None:
        self.prompt = prompt
        self.parser = parser
        self.sys_prompt = sys_prompt

    def generate_prompt(
        self,
        question: str,
        candidate_a: str,
        candidate_b: str,
    ) -> List:
        if self.sys_prompt is None:
            return [
                {
                    "name": "user",
                    "role": "user",
                    "content": self.prompt.format(
                        question=question,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        format=self.parser.format_instruction,
                    ),
                },
            ]
        else:
            return [
                {
                    "name": "system",
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "name": "user",
                    "role": "user",
                    "content": self.prompt.format(
                        question=question,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        format=self.parser.format_instruction,
                    ),
                },
            ]

    def parse_text(self, text: str) -> dict:
        result = {}
        try:
            result = self.parser.parse_to_dict(text)
        except Exception:
            pass
        result["raw"] = text
        return result
