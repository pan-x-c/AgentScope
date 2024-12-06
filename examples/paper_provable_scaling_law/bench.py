# -*- coding: utf-8 -*-
"""
Benchmark used in
"Simple and Provable Scaling Law for the
Test-Time Compute of Large Language Models"
"""
from __future__ import annotations
import json
import random
import argparse
from loguru import logger
from utils.dataset import Dataset
from utils.cache import Cache
from utils.worker import MixedGenerator, MixedJudge, Generator, Judge
from competition import Competition

import agentscope
from agentscope.server import RpcAgentServerLauncher


def run(
    generator: MixedGenerator,
    competition: Competition,
    dataset: Dataset,
    competition_config: dict,
    cache: Cache,
) -> None:
    """Run the benchmark."""
    max_pending_num = 4
    step_num = 1
    futures = []
    for question in dataset:
        if len(futures) > max_pending_num:
            for _ in range(step_num):
                futures.pop(0).result()
        futures.append(
            generator.generate(
                question=question,
                n=competition_config["candidate_num"],
            ),
        )
    for f in futures:
        f.result()
    logger.info("Generation finished")

    futures = []
    for question in dataset:
        if len(futures) > max_pending_num:
            for _ in range(step_num):
                futures.pop(0).result()
        futures.append(
            competition.competition(
                question=question,
                candidates=cache.load_generation(
                    question["id"],
                    question["category"],
                )[: competition_config["candidate_num"]],
                **competition_config,
            ),
        )
    for f in futures:
        f.result()
    logger.info("Knockout finished")


def main(conf: dict) -> None:
    """Main entry point."""
    dataset = Dataset.from_dict(conf["dataset"])
    cache = Cache(
        project_name=conf["project"],
        dataset_name=conf["dataset"]["name"],
        job_name=conf["job"],
    )
    worker_launchers = [RpcAgentServerLauncher(**s) for s in conf["servers"]]
    for launcher in worker_launchers:
        launcher.launch()
    master_launcher = RpcAgentServerLauncher()
    master_launcher.launch()
    generators = [
        Generator.from_dict(w).to_dist(
            worker_launchers[i % len(worker_launchers)].host,
            worker_launchers[i % len(worker_launchers)].port,
        )
        for i, w in enumerate(conf["generate"]["workers"])
    ]
    gen = MixedGenerator(
        generators=generators,
        cache=cache,
        to_dist={
            "host": master_launcher.host,
            "port": master_launcher.port,
        },
    )
    judge = MixedJudge(
        judges=[
            Judge.from_dict(w).to_dist(
                worker_launchers[i % len(worker_launchers)].host,
                worker_launchers[i % len(worker_launchers)].port,
            )
            for i, w in enumerate(conf["judge"]["workers"])
        ],
        cache=cache,
        to_dist={
            "host": master_launcher.host,
            "port": master_launcher.port,
        },
    )
    competition = Competition(
        judge,
        cache,
        to_dist={
            "host": master_launcher.host,
            "port": master_launcher.port,
        },
    )
    run(
        gen,
        competition,
        dataset,
        conf["competition"],
        cache,
    )
    master_launcher.shutdown()
    for launcher in worker_launchers:
        launcher.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    agentscope.init(
        project=config["project"],
        model_configs=config["models"],
    )
    main(config)
