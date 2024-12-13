# -*- coding: utf-8 -*-
import os
from typing import List
from matplotlib import pyplot as plt
from .cache import Cache


FIGURE_DIR = os.path.join(os.path.dirname(__file__), "imgs")


class KnockoutFigureDrawer:
    @classmethod
    def draw_acc(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ):
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_knockout_stats(
                n=config["n"],
                k=config["k"],
                categories=categories,
            )
            for config in configs
        ]
        for category in categories:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i, stat in enumerate(stats):
                ax.plot(
                    stat[category]["acc"].keys(),
                    stat[category]["acc"].values(),
                    label=configs[i]["label"],
                    marker=configs[i]["marker"],
                    color=configs[i]["color"],
                )
            ax.set_title(f"{dataset_name}: {category}")
            ax.grid(
                True,
                linestyle="dashed",
                linewidth=1,
                color="gray",
                alpha=0.5,
            )
            ax.set_xlabel("N")
            ax.set_ylabel("Accuracy")
            ax.legend(
                ncol=2,
                handlelength=1.8,
                handletextpad=0.2,
                labelspacing=0.2,
                columnspacing=0.3,
                loc="lower center",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(figure_dir, f"acc_{dataset_name}_{category}.pdf"),
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.savefig(
                os.path.join(figure_dir, f"acc_{dataset_name}_{category}.png"),
                bbox_inches="tight",
                pad_inches=0.02,
            )

    @classmethod
    def draw_p_cmp(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ):
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        run_stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_knockout_stats(
                n=config["n"],
                k=config["k"],
                categories=categories,
            )
            for config in configs
        ]
        for i, stats in enumerate(run_stats):
            for category in categories:
                all_correct_cnt = 0
                all_wrong_cnt = 0
                fig, ax = plt.subplots(figsize=(3.5, 3))
                right_p_gens = []
                right_p_cmps = []
                wrong_p_gens = []
                wrong_p_cmps = []
                for qid, stat in stats[category]["details"].items():
                    if stat["cmp"]["valid"] > 0:
                        if stat["acc"][str(configs[i]["n"])] == 1:
                            right_p_gens.append(stat["acc"]["1"])
                            right_p_cmps.append(stat["cmp"]["p_cmp"])
                        else:
                            wrong_p_gens.append(stat["acc"]["1"])
                            wrong_p_cmps.append(stat["cmp"]["p_cmp"])
                    if stat["acc"]["1"] == 0:
                        all_wrong_cnt += 1
                    if stat["acc"]["1"] == 1:
                        all_correct_cnt += 1
                ax.scatter(
                    right_p_gens,
                    right_p_cmps,
                    label=configs[i]["label"],
                    alpha=0.6,
                    color=configs[i]["color"],
                )
                ax.scatter(
                    wrong_p_gens,
                    wrong_p_cmps,
                    label=configs[i]["label"],
                    alpha=0.6,
                    color=configs[i]["color"],
                    marker="x",
                )
                above_count = sum(
                    1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp > 0.5
                )
                below_count = sum(
                    1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp <= 0.5
                )
                ax.set_title(
                    f"({configs[i]['label']}) {dataset_name}: {category}",
                )
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.10, 1.10)
                ax.set_xlabel("$P_{gen}$")
                ax.set_ylabel("$P_{comp}$")
                ax.axhline(
                    y=0.5,
                    color="black",
                    linestyle="dotted",
                    linewidth=1.0,
                )
                ax.grid(
                    True,
                    linestyle="dashed",
                    linewidth=1,
                    color="gray",
                    alpha=0.5,
                )
                ax.text(
                    -0.05,
                    1.03,
                    "#[$P_{comp}>0.5$] = " + str(above_count),
                    fontsize=9,
                    verticalalignment="center",
                )
                ax.text(
                    -0.05,
                    -0.05,
                    "#[$P_{comp}≤0.5$] = " + str(below_count),
                    fontsize=9,
                    verticalalignment="center",
                )

                ax.text(
                    -0.05,
                    -0.33,
                    "#[$P_{gen}$=0] = " + str(all_wrong_cnt),
                    fontsize=9,
                    verticalalignment="center",
                )
                ax.text(
                    0.65,
                    -0.33,
                    "#[$P_{gen}$=1] = " + str(all_correct_cnt),
                    fontsize=9,
                    verticalalignment="center",
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        figure_dir,
                        f"cmp_{configs[i]['label']}_{category}.pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.savefig(
                    os.path.join(
                        figure_dir,
                        f"cmp_{configs[i]['label']}_{category}.png",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )


class UCBFigureDrawer:
    # TODO: merge into KnockoutFigureDrawer
    @classmethod
    def draw_acc(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ):
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_ucb_stats(
                n=config["n"],
                k=config["k"],
                t=config["t"],
                categories=categories,
            )
            for config in configs
        ]
        for category in categories:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i, stat in enumerate(stats):
                ax.plot(
                    stat[category]["acc"].keys(),
                    stat[category]["acc"].values(),
                    label=configs[i]["label"],
                    marker=configs[i]["marker"],
                    color=configs[i]["color"],
                )
            ax.set_title(f"{dataset_name}: {category}")
            ax.grid(
                True,
                linestyle="dashed",
                linewidth=1,
                color="gray",
                alpha=0.5,
            )
            ax.set_xlabel("T")
            ax.set_ylabel("Accuracy")
            ax.legend(
                ncol=2,
                handlelength=1.8,
                handletextpad=0.2,
                labelspacing=0.2,
                columnspacing=0.3,
                loc="lower center",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    figure_dir,
                    f"acc_ucb_{dataset_name}_{category}.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.savefig(
                os.path.join(
                    figure_dir,
                    f"acc_ucb_{dataset_name}_{category}.png",
                ),
                bbox_inches="tight",
                pad_inches=0.02,
            )

    @classmethod
    def draw_p_cmp(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ):
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        run_stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_ucb_stats(
                n=config["n"],
                k=config["k"],
                t=config["t"],
                categories=categories,
            )
            for config in configs
        ]
        for i, stats in enumerate(run_stats):
            for category in categories:
                all_correct_cnt = 0
                all_wrong_cnt = 0
                fig, ax = plt.subplots(figsize=(3.5, 3))
                right_p_gens = []
                right_p_cmps = []
                wrong_p_gens = []
                wrong_p_cmps = []
                for qid, stat in stats[category]["details"].items():
                    if stat["cmp"]["valid"] > 0:
                        if stat["acc"][str(configs[i]["t"])] == 1:
                            right_p_gens.append(stat["acc"]["avg"])
                            right_p_cmps.append(stat["cmp"]["p_cmp"])
                        else:
                            wrong_p_gens.append(stat["acc"]["avg"])
                            wrong_p_cmps.append(stat["cmp"]["p_cmp"])
                    if stat["acc"]["avg"] == 0:
                        all_wrong_cnt += 1
                    if stat["acc"]["avg"] == 1:
                        all_correct_cnt += 1
                ax.scatter(
                    right_p_gens,
                    right_p_cmps,
                    label=configs[i]["label"],
                    alpha=0.6,
                    color=configs[i]["color"],
                )
                ax.scatter(
                    wrong_p_gens,
                    wrong_p_cmps,
                    label=configs[i]["label"],
                    alpha=0.6,
                    color=configs[i]["color"],
                    marker="x",
                )
                above_count = sum(
                    1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp > 0.5
                )
                below_count = sum(
                    1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp <= 0.5
                )
                ax.set_title(
                    f"({configs[i]['label']}) {dataset_name}: {category}",
                )
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.10, 1.10)
                ax.set_xlabel("$P_{gen}$")
                ax.set_ylabel("$P_{comp}$")
                ax.axhline(
                    y=0.5,
                    color="black",
                    linestyle="dotted",
                    linewidth=1.0,
                )
                ax.grid(
                    True,
                    linestyle="dashed",
                    linewidth=1,
                    color="gray",
                    alpha=0.5,
                )
                ax.text(
                    -0.05,
                    1.03,
                    "#[$P_{comp}>0.5$] = " + str(above_count),
                    fontsize=9,
                    verticalalignment="center",
                )
                ax.text(
                    -0.05,
                    -0.05,
                    "#[$P_{comp}≤0.5$] = " + str(below_count),
                    fontsize=9,
                    verticalalignment="center",
                )

                ax.text(
                    -0.05,
                    -0.33,
                    "#[$P_{gen}$=0] = " + str(all_wrong_cnt),
                    fontsize=9,
                    verticalalignment="center",
                )
                ax.text(
                    0.65,
                    -0.33,
                    "#[$P_{gen}$=1] = " + str(all_correct_cnt),
                    fontsize=9,
                    verticalalignment="center",
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        figure_dir,
                        f"cmp_ucb_{configs[i]['label']}_{category}.pdf",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.savefig(
                    os.path.join(
                        figure_dir,
                        f"cmp_ucb_{configs[i]['label']}_{category}.png",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
