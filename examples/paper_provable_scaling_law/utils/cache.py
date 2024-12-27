# -*- coding: utf-8 -*-
"""Cache module."""
import os
import json
from typing import List
from loguru import logger

DEFAULT_CATEGORY = "all"


class Cache:
    """A cache for storing and loading data.
    The cache directory is
    `output/<project_name>/<job_name>/`.
    """

    def __init__(
        self,
        project_name: str,
        job_name: str,
        config: dict = None,
    ) -> None:
        self.project_name = project_name
        self.job_name = job_name
        self.cache_root_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "output",
        )
        self.run_dir = os.path.join(
            self.cache_root_dir,
            self.project_name,
            self.job_name,
        )
        self.generation_dir = os.path.join(
            self.run_dir,
            "generation",
        )
        self.comparsion_dir = os.path.join(
            self.run_dir,
            "comparison",
        )
        self.competition_dir = os.path.join(
            self.run_dir,
            "competition",
        )
        if config:
            if "generation" in config:
                self.generation_dir = os.path.join(
                    self.cache_root_dir,
                    config["generation"]["project"],
                    config["generation"]["job"],
                    "generation",
                )
            if "comparison" in config:
                self.comparsion_dir = os.path.join(
                    self.cache_root_dir,
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
        """Save generation statistics to
        `<run_dir>/generation/stats/<category>/<instance_id>.json`
        """
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
        """Save generated candidates to
        `<run_dir>/generation/candidates/<category>/<instance_id>.jsonl`
        """
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
        """Load generated candidates from
        `<run_dir>/generation/candidates/<category>/<instance_id>.jsonl`
        """
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
        """Save pairwise comparison details to
        `<run_dir>/comparison/pairwise/<category>/<instance_id>/<cid_a>-<cid_b>.json`
        """
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
        """Load pairwise comparison details from
        `<run_dir>/comparison/pairwise/<category>/<instance_id>/<cid_a>-<cid_b>.json`
        """
        pairwise_file = os.path.join(
            self.comparsion_dir,
            "pairwise",
            category,
            str(instance_id),
            f"{cid_a}-{cid_b}.json",
        )
        if not os.path.exists(pairwise_file):
            return {}
        try:
            return json.load(open(pairwise_file, "r", encoding="utf-8"))
        except Exception as e:
            logger.error(
                f"Failed to load pairwise comparison for {instance_id}",
            )
            logger.error(e)
            return {}

    def save_competition(
        self,
        detail: dict,
        competition_type: str,
        instance_id: str,
        category: str,
        suffix: str,
    ) -> None:
        """Save competition details to
        `<run_dir>/competition/<competition_type>/<category>/<instance_id>_<suffix>.json`
        """
        competition_dir = os.path.join(
            self.competition_dir,
            competition_type,
            category,
        )
        os.makedirs(competition_dir, exist_ok=True)
        with open(
            os.path.join(competition_dir, f"{instance_id}_{suffix}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

    def save_competition_stats(
        self,
        stats: dict,
        competition_type: str,
        category: str,
        suffix: str,
    ) -> None:
        """Save competition statistics to
        `<run_dir>/competition/<competition_type>_stats/<category>_<suffix>.json`
        """
        competition_stats_dir = os.path.join(
            self.competition_dir,
            f"{competition_type}_stats",
        )
        os.makedirs(competition_stats_dir, exist_ok=True)
        with open(
            os.path.join(
                competition_stats_dir,
                f"{category}_{suffix}.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def load_competition(
        self,
        instance_id: str,
        competition_type: str,
        category: str,
        suffix: str,
    ) -> dict:
        """
        Load competition details from
        `<run_dir>/competition/<competition_type>/<category>/<instance_id>_<suffix>.json`
        """
        competition_file = os.path.join(
            self.competition_dir,
            competition_type,
            category,
            f"{instance_id}_{suffix}.json",
        )
        if not os.path.exists(competition_file):
            return {}
        return json.load(open(competition_file, "r", encoding="utf-8"))

    def load_competition_stats(
        self,
        competition_type: str,
        categories: List[str],
        suffix: str,
    ) -> dict:
        """
        Load competition statistics from
        `<run_dir>/competition/<competition_type>_stats/<category>_<suffix>.json`
        """
        result = {}
        for category in categories:
            competition_stats_file = os.path.join(
                self.competition_dir,
                f"{competition_type}_stats",
                f"{category}_{suffix}.json",
            )
            result[category] = json.load(
                open(competition_stats_file, "r", encoding="utf-8"),
            )
        return result
