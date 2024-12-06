# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from typing import List

from agentscope.rpc import async_func, RpcMeta
from agentscope.manager import ModelManager

from .prompter import GenerationPrompter, ComparisonPrompter
from .cache import Cache
from .parser import (
    MultiTagsParser,
    TagParser,
    PairWiseParser,
    MMLUProParser,
)


class Generator(metaclass=RpcMeta):
    def __init__(
        self,
        prompter: GenerationPrompter,
        model_config: str,
    ) -> None:
        self.model = ModelManager.get_instance().get_model_by_config_name(
            model_config,
        )
        self.prompter = prompter

    @classmethod
    def from_dict(cls, config: dict) -> Generator:
        if config["type"] == "mmlu_pro":
            return MMLUProGenerator(config["model"])
        else:
            raise NotImplementedError

    @async_func
    def run(self, question: str) -> dict:
        prompt = self.prompter.generate_prompt(question=question)
        return self.prompter.parse_text(self.model(prompt).text)


class MixedGenerator(metaclass=RpcMeta):
    def __init__(
        self,
        generators: List[Generator],
        cache: Cache,
    ) -> None:
        self.generators = generators
        self.generator_num = len(self.generators)
        self.cache = cache

    @async_func
    def generate(self, question: dict, n: int) -> List[dict]:
        candidates = self.cache.load_generation(
            instance_id=question["id"],
            category=question.get("category", "all"),
        )
        diff = max(0, n - len(candidates))
        if diff == 0:
            return candidates
        futures = [
            self.generators[i % self.generator_num].run(
                question=question["question"],
            )
            for i in range(diff)
        ]
        candidates.extend([_.result() for _ in futures])
        self.cache.save_generation(
            candidates=candidates,
            instance_id=question["id"],
            category=question.get("category", "all"),
        )
        self.cache.save_generation_stats(
            stats=self.calculate_stats(question, candidates),
            instance_id=question["id"],
            category=question.get("category", "all"),
        )
        return candidates

    def calculate_stats(self, question: dict, candidates: List[dict]) -> dict:
        def _get_majority(candidates: list[dict]) -> str:
            votes = {}
            for c in candidates:
                if c["answer"] not in votes:
                    votes[c["answer"]] = 0
                else:
                    votes[c["answer"]] += 1
            return max(votes, key=votes.get)

        stats = {
            "target": question["answer"],
            "question": question["question"],
            "category": question["category"],
        }
        total = 0
        correct = 0
        for candidate in candidates:
            total += 1
            if candidate["answer"] == question["answer"]:
                correct += 1
        stats["acc"] = correct / total
        stats["total_cnt"] = total
        stats["correct_cnt"] = correct
        stats["majority"] = _get_majority(candidates)
        return stats


class Judge(metaclass=RpcMeta):
    def __init__(
        self,
        prompter: ComparisonPrompter,
        model_config: str,
    ) -> None:
        self.model = ModelManager.get_instance().get_model_by_config_name(
            model_config,
        )
        self.prompter = prompter

    @classmethod
    def from_dict(cls, config: dict) -> Judge:
        if config["type"] == "mmlu_pro":
            return MMLUProJudge(model_config=config["model"])
        else:
            raise NotImplementedError

    @async_func
    def run(self, question: str, candidate_a: str, candidate_b: str) -> dict:
        prompt = self.prompter.generate_prompt(
            question=question,
            candidate_a=candidate_b,
            candidate_b=candidate_a,
        )
        return self.prompter.parse_text(self.model(prompt).text)


class MixedJudge(metaclass=RpcMeta):
    def __init__(
        self,
        judges: List[Judge],
        cache: Cache,
    ):
        self.judges = judges
        self.judge_num = len(self.judges)
        self.cache = cache

    @async_func
    def pairwise_compare(
        self,
        question: dict,
        candidate_a: dict,
        candidate_b: dict,
        k: int,
    ) -> dict:
        cache_result = self.cache.load_pairwise_comparison(
            instance_id=question["id"],
            cid_a=candidate_a["cid"],
            cid_b=candidate_b["cid"],
            category=question.get("category", "all"),
        )
        cmp_num = cache_result.get("cmp_num", 0)
        k = max(0, k - cmp_num)
        if k == 0:
            return cache_result
        a_b = [
            self.judges[i % self.judge_num].run(
                question=question["question"],
                candidate_a=candidate_a["raw"],
                candidate_b=candidate_b["raw"],
            )
            for i in range(k // 2)
        ]
        b_a = [
            self.judges[i % self.judge_num].run(
                question=question["question"],
                candidate_a=candidate_b["raw"],
                candidate_b=candidate_a["raw"],
            )
            for i in range(k // 2)
        ]
        a_b = [_.result() for _ in a_b]
        b_a = [_.result() for _ in b_a]
        result = {
            "cmp_num": cmp_num + k,
            "score_a": cache_result.get("score_a", 0)
            + sum(_["winner"].get("a") for _ in a_b)
            + sum(_["winner"].get("b") for _ in b_a),
            "score_b": cache_result.get("score_b", 0)
            + sum(_["winner"].get("b") for _ in a_b)
            + sum(_["winner"].get("a") for _ in b_a),
            "a": candidate_a["cid"],
            "b": candidate_b["cid"],
            "cmp_a_b": cache_result.get("cmp_a_b", []) + a_b,
            "cmp_b_a": cache_result.get("cmp_b_a", []) + b_a,
        }
        if result["score_a"] > result["score_b"]:
            result["winner"] = candidate_a["cid"]
        elif result["score_a"] < result["score_b"]:
            result["winner"] = candidate_b["cid"]
        else:
            result["winner"] = random.choice(
                [candidate_a["cid"], candidate_b["cid"]],
            )
        self.cache.save_pairwise_comparison(
            detail=result,
            instance_id=question["id"],
            cid_a=candidate_a["cid"],
            cid_b=candidate_b["cid"],
            category=question.get("category", "all"),
        )
        return result


class MMLUProGenerator(Generator):
    GEN_PROMPT = """# Question

{question}

# Output Format
```
{format}
```
"""

    def __init__(self, model_config: str):
        generation_prompter = GenerationPrompter(
            prompt=MMLUProGenerator.GEN_PROMPT,
            parser=MultiTagsParser(
                tags=[
                    TagParser("reason", "your step-by-step reasoning process"),
                    MMLUProParser("answer"),
                ],
            ),
            sys_prompt="Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.",
        )
        super().__init__(generation_prompter, model_config)


class MMLUProJudge(Judge):
    CMP_PROMPT = """---- QUESTION ----
{question}

---- Solution 1 ----
{candidate_a}

---- Solution 2 ----
{candidate_b}

---- OUTPUT FORMAT ----
```
{format}
```
"""

    def __init__(self, model_config: str):
        comparison_prompter = ComparisonPrompter(
            prompt=MMLUProJudge.CMP_PROMPT,
            parser=MultiTagsParser(
                tags=[
                    TagParser(
                        "compare",
                        "compare both candidate solutions step-by-step thoroughly, and double check if there are mistakes in either solution",
                    ),
                    PairWiseParser(),
                ],
            ),
            sys_prompt="You are an impartial Judge. Given a question and two candidate solutions, your task is to choose which solution answer the question better. Your judgment should be unbiased, without favoring either Solution 1 or 2.",
        )
        super().__init__(comparison_prompter, model_config)
