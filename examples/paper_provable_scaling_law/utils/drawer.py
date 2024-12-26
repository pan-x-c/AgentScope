# -*- coding: utf-8 -*-
"""Drawer for competition figures."""
import os
from collections import defaultdict
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from .cache import Cache


FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imgs")


class CompetitionFigureDrawer:
    """
    Draw competition figures.
    """

    @classmethod
    def _construct_suffix(cls, competition_type: str, config: dict) -> str:
        """Construct cache file suffix string."""
        if competition_type == "knockout":
            return f"{config['n']}_{config['k']}"
        elif competition_type == "league":
            return f"{config['n']}_{config['k']}_{config['m']}"
        elif competition_type == "lucb":
            return f"{config['n']}_{config['k']}_{config['t']}"
        else:
            raise ValueError(f"Unknown competition type: {competition_type}")

    @classmethod
    def _load_runs(
        cls,
        run_configs: List[dict],
        competition_type: str,
        categories: List[str],
    ) -> List[dict]:
        """Load cached run stats."""
        return [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=competition_type,
                categories=categories,
                suffix=cls._construct_suffix(competition_type, config),
            )
            for config in run_configs
        ]

    @classmethod
    def _draw_acc_line(
        cls,
        dataset_name: str,
        category: str,
        competition_type: str,
        lines: List[dict],
        figure_dir: str,
        x_label: str = "N",
    ) -> None:
        """Draw accuracy line plot."""
        _, ax = plt.subplots(figsize=(4, 3))
        for line in lines:
            ax.plot(
                line["acc"].keys(),
                line["acc"].values(),
                label=line["label"],
                marker=line["marker"],
                color=line["color"],
                linestyle=line.get("linestyle", "solid"),
            )
        ax.set_title(f"{dataset_name}: {category}")
        ax.grid(
            True,
            linestyle="dashed",
            linewidth=1,
            color="gray",
            alpha=0.5,
        )
        ax.set_xlabel(x_label)
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
                f"acc_{competition_type}_{dataset_name}_{category}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.savefig(
            os.path.join(
                figure_dir,
                f"acc_{competition_type}_{dataset_name}_{category}.png",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )

    @classmethod
    def draw_acc(
        cls,
        competition_type: str,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        """Draw the accuracy vs N/T line plot."""
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        stats = cls._load_runs(configs, competition_type, categories)
        # draw categories
        for category in categories:
            lines = []
            for i, stat in enumerate(stats):
                line = {"acc": stat[category]["acc"]}
                line.update(configs[i])
                lines.append(line)
            cls._draw_acc_line(
                dataset_name=dataset_name,
                category=category,
                competition_type=competition_type,
                lines=lines,
                figure_dir=figure_dir,
                x_label="T" if competition_type == "lucb" else "N",
            )
        # draw all
        all_lines = []
        for i, stat in enumerate(stats):
            all_cnt = 0
            all_acc = defaultdict(float)
            for category in categories:
                for k, v in stat[category]["acc"].items():
                    all_acc[k] += v * stat[category]["cnt"]
                all_cnt += stat[category]["cnt"]
            for k in all_acc:
                all_acc[k] /= all_cnt
            line = {"acc": all_acc}
            line.update(configs[i])
            all_lines.append(line)
        print(f"[ALL]: {all_lines}")
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            competition_type=competition_type,
            lines=all_lines,
            figure_dir=figure_dir,
            x_label="T" if competition_type == "lucb" else "N",
        )

    @classmethod
    def draw_majority_vote(
        cls,
        competition_type: str,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        """Draw the line plot of majority vote accuracy."""
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=competition_type,
                categories=categories,
                suffix=cls._construct_suffix(competition_type, config),
            )
            for config in configs
        ]
        # draw categories
        for category in categories:
            lines = []
            for i, stat in enumerate(stats):
                line = {"acc": stat[category]["acc"]}
                majority_line = {"acc": stat[category]["majority_acc"]}
                line.update(configs[i])
                majority_line.update(configs[i])
                majority_line["linestyle"] = "dotted"
                majority_line["marker"] = "s"
                majority_line["label"] += "(MV)"
                lines.append(line)
                lines.append(majority_line)
            cls._draw_acc_line(
                dataset_name=dataset_name,
                category=category,
                competition_type=competition_type,
                lines=lines,
                figure_dir=figure_dir,
            )
        # draw all
        all_lines = []
        for i, stat in enumerate(stats):
            all_cnt = 0
            all_acc = defaultdict(float)
            all_majority_acc = defaultdict(float)
            for category in categories:
                for k, v in stat[category]["acc"].items():
                    all_acc[k] += v * stat[category]["cnt"]
                for k, v in stat[category]["majority_acc"].items():
                    all_majority_acc[k] += v * stat[category]["cnt"]
                all_cnt += stat[category]["cnt"]
            for k in all_acc:
                all_acc[k] /= all_cnt
            for k in all_majority_acc:
                all_majority_acc[k] /= all_cnt
            line = {"acc": all_acc}
            majority_line = {"acc": all_majority_acc}
            line.update(configs[i])
            majority_line.update(configs[i])
            majority_line["linestyle"] = "dotted"
            majority_line["marker"] = "s"
            majority_line["label"] += "(MV)"
            all_lines.append(line)
            all_lines.append(majority_line)
        print(f"[ALL]: {all_lines}")
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            competition_type=competition_type,
            lines=all_lines,
            figure_dir=figure_dir,
        )

    @classmethod
    def draw_scatter(
        cls,
        dataset_name: str,
        category: str,
        competition_type: str,
        details: dict,
        config: dict,
        figure_dir: str,
        judge_field: str,
        threshold: float = 1.0,
    ) -> None:
        """Draw scatter plot of P_cmp and P_gen"""
        all_correct_cnt = 0
        all_wrong_cnt = 0
        right_p_gens = []
        right_p_cmps = []
        wrong_p_gens = []
        wrong_p_cmps = []
        for stat in details.values():
            if stat["cmp"]["valid"] > 0:
                if stat["acc"][judge_field] >= threshold:
                    right_p_gens.append(stat["acc"]["1"])
                    right_p_cmps.append(stat["cmp"]["p_cmp"])
                else:
                    wrong_p_gens.append(stat["acc"]["1"])
                    wrong_p_cmps.append(stat["cmp"]["p_cmp"])
            if stat["acc"]["1"] == 0:
                all_wrong_cnt += 1
            if stat["acc"]["1"] == 1:
                all_correct_cnt += 1
        fig = plt.figure(figsize=(4, 3))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax)
        ax.scatter(
            right_p_gens,
            right_p_cmps,
            15,
            label=config["label"],
            alpha=0.6,
            color=config["color"],
        )
        ax.scatter(
            wrong_p_gens,
            wrong_p_cmps,
            15,
            label=config["label"],
            alpha=0.6,
            color=config["color"],
            marker="x",
        )
        bins = np.linspace(-0.0, 1.0, 50)
        ax_hist.hist(
            right_p_cmps + wrong_p_cmps,
            bins=bins,
            orientation="horizontal",
            color=config["color"],
            alpha=0.6,
        )
        ax_hist.set_axis_off()
        above_count = sum(
            1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp > 0.5
        )
        below_count = sum(
            1 for p_cmp in right_p_cmps + wrong_p_cmps if p_cmp <= 0.5
        )
        ax.set_title(
            f"{dataset_name}: {category}",
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
                f"cmp_{competition_type}_{config['label']}_{category}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.savefig(
            os.path.join(
                figure_dir,
                f"cmp_{competition_type}_{config['label']}_{category}.png",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )

    @classmethod
    def draw_gen_cmp_scatter(
        cls,
        competition_type: str,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
        threshold: float = 1.0,
    ) -> None:
        """
        Draw scatter plot of P_cmp and P_gen
        """
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        run_stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=competition_type,
                categories=categories,
                suffix=cls._construct_suffix(competition_type, config),
            )
            for config in configs
        ]
        for i, stats in enumerate(run_stats):
            for category in categories:
                judge_field = (
                    str(configs[i]["n"])
                    if competition_type != "lucb"
                    else str(configs[i]["t"])
                )
                cls.draw_scatter(
                    dataset_name=dataset_name,
                    category=category,
                    competition_type=competition_type,
                    details=stats[category]["details"],
                    config=configs[i],
                    figure_dir=figure_dir,
                    judge_field=judge_field,
                    threshold=threshold,
                )

    @classmethod
    def draw_subset_acc(
        cls,
        competition_type: str,
        threshold: float,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        """Draw the accuracy of the subset that meet the condition."""
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        run_stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=competition_type,
                categories=categories,
                suffix=cls._construct_suffix(competition_type, config),
            )
            for config in configs
        ]
        # draw categories
        lines = []
        for i, run in enumerate(run_stats):
            line = {"acc": defaultdict(float), "cnt": 0}
            for category in categories:
                question_stats = run[category]["details"]
                for stat in question_stats.values():
                    if stat["cmp"]["p_cmp"] < threshold:
                        continue
                    line["cnt"] += 1
                    for k, v in stat["acc"].items():
                        line["acc"][k] += v
            for k in line["acc"]:
                line["acc"][k] /= line["cnt"]
            line.update(configs[i])
            lines.append(line)
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            lines=lines,
            competition_type=competition_type,
            figure_dir=figure_dir,
        )

    @classmethod
    def draw_acc_vs_m(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        competition_type = "league"
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        run_stats = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=competition_type,
                categories=categories,
                suffix=cls._construct_suffix(competition_type, config),
            )
            for config in configs
        ]
        # draw categories
        lines = []
        for i, run in enumerate(run_stats):
            line = {"acc": defaultdict(float), "cnt": 0}
            for category in categories:
                for k, v in run[category]["acc_vs_m"].items():
                    line["acc"][k] += v * run[category]["cnt"]
                line["cnt"] += run[category]["cnt"]
            for k in line["acc"]:
                line["acc"][k] /= line["cnt"]
            line.update(configs[i])
            lines.append(line)
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            lines=lines,
            competition_type=competition_type,
            figure_dir=figure_dir,
            x_label="M",
        )
