# -*- coding: utf-8 -*-
"""
Benchmark used in
"Simple and Provable Scaling Law for the
Test-Time Compute of Large Language Models"
"""
from __future__ import annotations
import json
import argparse
from loguru import logger
import agentscope
from agentscope.server import RpcAgentServerLauncher

from utils import get_dataset, get_generator, get_judge, get_competition
from utils.dataset import Dataset
from utils.cache import Cache
from utils.worker import MixedGenerator, MixedJudge
from utils.competition import Competition


def run_generation(
    generator: MixedGenerator,
    dataset: Dataset,
    pending_task_num: int = 4,
    step_num: int = 1,
) -> dict:
    """Run the generation."""
    futures = []
    for question in dataset:
        if len(futures) > pending_task_num:
            for _ in range(step_num):
                futures.pop(0).result()
        futures.append(
            generator.generate(
                question=question,
            ),
        )
    for f in futures:
        f.result()
    logger.info("Generation finished")


def run_competition(
    competition: Competition,
    dataset: Dataset,
    cache: Cache,
    pending_task_num: int = 2,
    step_num: int = 1,
) -> dict:
    """Run the competition."""
    futures = []
    for question in dataset:
        if len(futures) > pending_task_num:
            for _ in range(step_num):
                futures.pop(0).result()
        futures.append(
            competition.competition(
                question=question,
                candidates=cache.load_generation(
                    question["id"],
                    question["category"],
                ),
            ),
        )
    for f in futures:
        f.result()
    competition.calculate_stats(dataset)
    logger.info("Competition finished")


def main(conf: dict) -> None:
    """Main entry point."""
    dataset = get_dataset(conf["dataset"])
    cache = Cache(
        project_name=conf["project"],
        job_name=conf["job"],
        config=conf.get("cache", None),
    )
    worker_launchers = [RpcAgentServerLauncher(**s) for s in conf["servers"]]
    for launcher in worker_launchers:
        launcher.launch()
    master_launcher = RpcAgentServerLauncher()
    master_launcher.launch()
    if config.get("generation", None):
        generators = [
            get_generator(w).to_dist(
                worker_launchers[i % len(worker_launchers)].host,
                worker_launchers[i % len(worker_launchers)].port,
            )
            for i, w in enumerate(conf["generation"]["workers"])
        ]
        gen = MixedGenerator(
            generators=generators,
            cache=cache,
            candidate_num=conf["generation"]["candidate_num"],
            to_dist={
                "host": master_launcher.host,
                "port": master_launcher.port,
            },
        )
        run_generation(
            gen,
            dataset,
        )
    judge = MixedJudge(
        judges=[
            get_judge(w).to_dist(
                host=worker_launchers[i % len(worker_launchers)].host,
                port=worker_launchers[i % len(worker_launchers)].port,
            )
            for i, w in enumerate(conf["judgement"]["workers"])
        ],
        cache=cache,
        to_dist={
            "host": master_launcher.host,
            "port": master_launcher.port,
        },
    )
    competition = get_competition(
        config=conf["competition"],
        judge=judge,
        cache=cache,
        to_dist={
            "host": master_launcher.host,
            "port": master_launcher.port,
        },
    )
    run_competition(
        competition,
        dataset,
        cache,
    )
    master_launcher.shutdown()
    for launcher in worker_launchers:
        launcher.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()
    config = json.load(open(args.config, "r", encoding="utf-8"))
    agentscope.init(
        project=config["project"],
        model_configs=config["models"],
        use_monitor=False,
        save_code=False,
    )
    main(config)
