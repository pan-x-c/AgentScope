# -*- coding: utf-8 -*-
"""Cache module."""
import os
import json
from typing import List
from loguru import logger

DEFAULT_CATEGORY = "all"


class Cache:
    """A cache for storing and loading data"""

    def __init__(
        self,
        project_name: str,
        job_name: str,
        config: dict = None,
    ) -> None:
        self.project_name = project_name
        self.job_name = job_name
        self.data_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            self.project_name,
            self.job_name,
        )
        self.generation_dir = os.path.join(
            self.data_dir,
            "generation",
        )
        self.comparsion_dir = os.path.join(
            self.data_dir,
            "comparison",
        )
        self.competition_dir = os.path.join(
            self.data_dir,
            "competition",
        )
        if config:
            if "generation" in config:
                self.generation_dir = os.path.join(
                    os.path.dirname(__file__),
                    "output",
                    config["generation"]["project"],
                    config["generation"]["job"],
                    "generation",
                )
            if "comparison" in config:
                self.comparsion_dir = os.path.join(
                    os.path.dirname(__file__),
                    "output",
                    config["comparison"]["project"],
                    config["comparison"]["job"],
                    "comparison",
                )
        os.makedirs(self.generation_dir, exist_ok=True)
        os.makedirs(self.comparsion_dir, exist_ok=True)
        os.makedirs(self.competition_dir, exist_ok=True)

    def save_generation_stats(
        self,
        stats: dict,
        instance_id: str,
        category: str = DEFAULT_CATEGORY,
    ) -> None:
        stats_dir = os.path.join(self.generation_dir, "stats", category)
        os.makedirs(stats_dir, exist_ok=True)
        json.dump(
            stats,
            open(
                os.path.join(
                    stats_dir,
                    f"{instance_id}.json",
                ),
                "w",
                encoding="utf-8",
            ),
            ensure_ascii=False,
            indent=2,
        )

    def save_generation(
        self,
        candidates: List[dict],
        instance_id: str,
        category: str = DEFAULT_CATEGORY,
    ) -> None:
        candidates_dir = os.path.join(
            self.generation_dir,
            "candidates",
            category,
        )
        os.makedirs(candidates_dir, exist_ok=True)
        with open(
            os.path.join(
                candidates_dir,
                f"{instance_id}.jsonl",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            for i, c in enumerate(candidates):
                c["cid"] = i
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def load_generation(
        self,
        instance_id: str,
        category: str = DEFAULT_CATEGORY,
    ) -> List[dict]:
        candidates_file = os.path.join(
            self.generation_dir,
            "candidates",
            category,
            f"{instance_id}.jsonl",
        )
        if not os.path.exists(candidates_file):
            logger.info(f"No generation found for instance {instance_id}")
            return []
        with open(candidates_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()]

    def save_pairwise_comparison(
        self,
        detail: dict,
        instance_id: str,
        cid_a: str,
        cid_b: str,
        category: str = DEFAULT_CATEGORY,
    ) -> None:
        pairwise_dir = os.path.join(
            self.comparsion_dir,
            "pairwise",
            category,
            str(instance_id),
        )
        os.makedirs(pairwise_dir, exist_ok=True)
        with open(
            os.path.join(pairwise_dir, f"{cid_a}-{cid_b}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

    def load_pairwise_comparison(
        self,
        instance_id: str,
        cid_a: str,
        cid_b: str,
        category: str = DEFAULT_CATEGORY,
    ) -> dict:
        pairwise_file = os.path.join(
            self.comparsion_dir,
            "pairwise",
            category,
            str(instance_id),
            f"{cid_a}-{cid_b}.json",
        )
        if not os.path.exists(pairwise_file):
            return {}
        return json.load(open(pairwise_file, "r", encoding="utf-8"))

    def save_knockout(
        self,
        detail: dict,
        instance_id: str,
        n: int,
        k: int,
        category: str = DEFAULT_CATEGORY,
    ) -> None:
        knockout_dir = os.path.join(self.competition_dir, "knockout", category)
        os.makedirs(knockout_dir, exist_ok=True)
        with open(
            os.path.join(knockout_dir, f"{instance_id}_{n}_{k}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

    def load_knockout(
        self,
        instance_id: str,
        n: int,
        k: int,
        category: str = DEFAULT_CATEGORY,
    ) -> dict:
        knockout_file = os.path.join(
            self.competition_dir,
            "knockout",
            category,
            f"{instance_id}_{n}_{k}.json",
        )
        if not os.path.exists(knockout_file):
            return {}
        return json.load(open(knockout_file, "r", encoding="utf-8"))


    def save_knockout_stats(
        self,
        stats: dict,
        n: int,
        k: int,
    ) -> None:
        knockout_stats_dir = os.path.join(
            self.competition_dir,
            "knockout_stats",
        )
        os.makedirs(knockout_stats_dir, exist_ok=True)
        with open(
            os.path.join(
                knockout_stats_dir,
                f"{n}_{k}.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def load_knockout_stats(
        self,
        n: int,
        k: int,
    ) -> dict:
        knockout_stats_file = os.path.join(
            self.competition_dir,
            "knockout_stats",
            f"{n}_{k}.json",
        )
        return {} if not os.path.exists(knockout_stats_file) else json.load(
            open(knockout_stats_file, "r", encoding="utf-8")
        )
        