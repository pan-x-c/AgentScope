# -*- coding: utf-8 -*-
"""Drawer for competition figures."""
import os
import json
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
        n_col: int = 3,
    ) -> None:
        """Draw accuracy line plot."""
        _, ax = plt.subplots(figsize=(4, 3))
        for line in lines:
            if competition_type != "lucb":
                xs = line["acc"].keys()
            else:
                xs = list(map(int, line["acc"].keys()))
            ax.plot(
                xs,
                line["acc"].values(),
                label=line["label"],
                marker=line["marker"],
                color=line["color"],
                linestyle=line.get("linestyle", "solid"),
                markevery=line.get("markevery", 1),
            )
            if "hline" in line:
                ax.axhline(
                    y=line["hline"],
                    linestyle="dotted",
                    color=line["color"],
                    linewidth=2.0,
                    alpha=0.8,
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
            ncol=n_col,
            handlelength=1.6,
            handletextpad=0.2,
            labelspacing=0.2,
            columnspacing=0.3,
            loc="best",
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
        print(f"[ALL]: {json.dumps(all_lines, ensure_ascii=False)}")
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
            majority_lines = []
            for i, stat in enumerate(stats):
                line = {"acc": stat[category]["acc"]}
                majority_line = {"acc": stat[category]["majority_acc"]}
                line.update(configs[i])
                majority_line.update(configs[i])
                majority_line["linestyle"] = "dotted"
                majority_line["label"] += " (MV)"
                lines.append(line)
                majority_lines.append(majority_line)
            cls._draw_acc_line(
                dataset_name=dataset_name,
                category=category,
                competition_type=competition_type,
                lines=lines + majority_lines,
                figure_dir=figure_dir,
                n_col=2,
            )
        # draw all
        all_lines = []
        all_majority_lines = []
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
            majority_line["label"] += "(MV)"
            all_lines.append(line)
            all_majority_lines.append(majority_line)
        print(f"[ALL]: {all_lines}")
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            competition_type=competition_type,
            lines=all_lines + all_majority_lines,
            figure_dir=figure_dir,
            n_col=2,
        )

    @classmethod
    def calculate_scatter_state(
        cls,
        competition_type: str,
        details: list,
    ) -> dict:
        """Calculate the state for scatter plot."""
        if competition_type == "league":
            middle = 0.0
        else:
            middle = 0.5
        all_correct_cnt = 0
        all_wrong_cnt = 0
        right_p_gens = []
        right_p_cmps = []
        wrong_p_gens = []
        wrong_p_cmps = []
        above_cnt = 0
        below_cnt = 0
        for stat in details:
            p_gen = next(iter(stat["acc"].values()))
            if competition_type == "league":
                y_value = (
                    stat["cmp"]["max_correct_win_rate"]
                    - stat["cmp"]["max_incorrect_win_rate"]
                )
            else:
                y_value = stat["cmp"]["p_cmp"]
            if stat["cmp"]["valid"] > 0:
                final_p_gen = next(reversed(stat["acc"].values()))
                if final_p_gen >= 0.5:
                    right_p_gens.append(p_gen)
                    right_p_cmps.append(y_value)
                else:
                    wrong_p_gens.append(p_gen)
                    wrong_p_cmps.append(y_value)
                if y_value > middle:
                    above_cnt += 1
                else:
                    below_cnt += 1
            if p_gen == 0:
                all_wrong_cnt += 1
            if p_gen == 1:
                all_correct_cnt += 1

        return {
            "right_p_gens": right_p_gens,
            "right_p_cmps": right_p_cmps,
            "wrong_p_gens": wrong_p_gens,
            "wrong_p_cmps": wrong_p_cmps,
            "above_cnt": above_cnt,
            "below_cnt": below_cnt,
            "all_correct_cnt": all_correct_cnt,
            "all_wrong_cnt": all_wrong_cnt,
        }

    @classmethod
    def _draw_scatter_plot(
        cls,
        dataset_name: str,
        category: str,
        competition_type: str,
        data: dict,
        config: dict,
        figure_dir: str,
        bottom: float = 0.0,
        top: float = 1.0,
        x_label: str = r"$\hat{P}_{gen}$",  # noqa: W605
        y_label: str = r"$\hat{P}_{comp}$",  # noqa: W605
    ) -> None:
        """Draw scatter plot of P_cmp and P_gen"""
        all_correct_cnt = data["all_correct_cnt"]
        all_wrong_cnt = data["all_wrong_cnt"]
        right_p_gens = data["right_p_gens"]
        right_p_cmps = data["right_p_cmps"]
        wrong_p_gens = data["wrong_p_gens"]
        wrong_p_cmps = data["wrong_p_cmps"]
        fig = plt.figure(figsize=(4, 3))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.0)
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax)
        ax.grid(
            True,
            linestyle="dashed",
            linewidth=0.8,
            color="#ccc",
            alpha=0.6,
        )
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
        bins = np.linspace(bottom, top, 50)
        ax_hist.hist(
            right_p_cmps + wrong_p_cmps,
            bins=bins,
            orientation="horizontal",
            color=config["color"],
            alpha=0.6,
        )
        ax_hist.set_axis_off()
        ax.set_title(
            f"{dataset_name}: {category}",
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(
            bottom - 0.10 * (top - bottom),
            top + 0.10 * (top - bottom),
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        middle = (top + bottom) / 2
        ax.axhline(
            y=middle,
            color="black",
            linestyle="dotted",
            linewidth=1.0,
        )
        ax.text(
            -0.05,
            top + 0.03 * (top - bottom),
            f"#[{y_label}$>{middle}$] = " + str(data["above_cnt"]),
            fontsize=9,
            verticalalignment="center",
        )
        ax.text(
            -0.05,
            bottom - 0.05 * (top - bottom),
            f"#[{y_label}$≤{middle}$] = " + str(data["below_cnt"]),
            fontsize=9,
            verticalalignment="center",
        )

        ax.text(
            -0.25,
            bottom - 0.33 * (top - bottom),
            f"#[{x_label}$=0$] = " + str(all_wrong_cnt),
            fontsize=9,
            verticalalignment="center",
        )
        ax.text(
            0.75,
            bottom - 0.33 * (top - bottom),
            f"#[{x_label}$=1$] = " + str(all_correct_cnt),
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
            run_details = []
            for category in categories:
                run_details.extend(stats[category]["details"].values())
                cls._draw_scatter_plot(
                    dataset_name=dataset_name,
                    category=category,
                    competition_type=competition_type,
                    data=cls.calculate_scatter_state(
                        details=stats[category]["details"].values(),
                        competition_type=competition_type,
                    ),
                    config=configs[i],
                    figure_dir=figure_dir,
                    bottom=0.0 if competition_type != "league" else -1.0,
                    top=1.0,
                    y_label=(
                        r"$\hat{P}_{comp}$"
                        if competition_type != "league"
                        else r"$\hat{\Delta}$"
                    ),
                )
            cls._draw_scatter_plot(
                dataset_name=dataset_name,
                category="all",
                competition_type=competition_type,
                data=cls.calculate_scatter_state(
                    details=run_details,
                    competition_type=competition_type,
                ),
                config=configs[i],
                figure_dir=figure_dir,
                bottom=0.0 if competition_type != "league" else -1.0,
                top=1.0,
                y_label=(
                    r"$\hat{P}_{comp}$"
                    if competition_type != "league"
                    else r"$\hat{\Delta}$"
                ),
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
            cnt = 0
            acc = defaultdict(float)
            for category in categories:
                question_stats = run[category]["details"]
                for stat in question_stats.values():
                    if stat["cmp"]["p_cmp"] < threshold:
                        continue
                    cnt += 1
                    for k, v in stat["acc"].items():
                        acc[k] += v
            for k in acc:
                acc[k] /= cnt
            line = {
                "acc": acc,
                "cnt": cnt,
            }
            line.update(configs[i])
            lines.append(line)
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all (subset)",
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
        """Draw acc vs m for league competition."""
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
            acc = defaultdict(float)
            cnt = 0
            for category in categories:
                for k, v in run[category]["acc_vs_m"].items():
                    acc[k] += v * run[category]["cnt"]
                cnt += run[category]["cnt"]
            for k in acc:
                acc[k] /= cnt
            line = {"acc": acc, "cnt": cnt}
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

    @classmethod
    def draw_pool_acc(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        """Draw pool acc for lucb competition."""
        competition_type = "lucb"
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        os.makedirs(figure_dir, exist_ok=True)
        stats = cls._load_runs(configs, competition_type, categories)
        # draw categories
        for category in categories:
            lines = []
            for i, stat in enumerate(stats):
                line = {"acc": stat[category]["pool_acc"]}
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
                for k, v in stat[category]["pool_acc"].items():
                    all_acc[k] += v * stat[category]["cnt"]
                all_cnt += stat[category]["cnt"]
            for k in all_acc:
                all_acc[k] /= all_cnt
            line = {"acc": all_acc}
            line.update(configs[i])
            all_lines.append(line)
        print(f"[ALL]: {json.dumps(all_lines)}")
        cls._draw_acc_line(
            dataset_name=dataset_name,
            category="all",
            competition_type=competition_type,
            lines=all_lines,
            figure_dir=figure_dir,
            x_label="T" if competition_type == "lucb" else "N",
        )

    @classmethod
    def _draw_pie(
        cls,
        dataset_name: str,
        category: str,
        values: List,
        labels: List,
        figure_dir: str,
    ) -> None:
        os.makedirs(figure_dir, exist_ok=True)
        _, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#99FF99", "#FF9999", "#FFCC99", "#66B3FF"],
        )
        ax.set_title(f"{dataset_name}: {category}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                f"pie_{category}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.savefig(
            os.path.join(
                figure_dir,
                f"pie_{category}.png",
            ),
            bbox_inches="tight",
            pad_inches=0.02,
        )

    @classmethod
    def draw_diff_pie(
        cls,
        dataset_name: str,
        categories: List[str],
        configs: List[dict],
        sub_dir: str = "default",
    ) -> None:
        """Draw pie chart for different competition."""
        assert len(configs) == 2
        figure_dir = os.path.join(FIGURE_DIR, sub_dir)
        run_a, run_b = [
            Cache(
                project_name=config["project"],
                job_name=config["job"],
            ).load_competition_stats(
                competition_type=config["competition_type"],
                categories=categories,
                suffix=cls._construct_suffix(
                    config["competition_type"],
                    config,
                ),
            )
            for config in configs
        ]
        both_right = 0
        both_wrong = 0
        a_right_b_wrong = 0
        a_wrong_b_right = 0
        arbw_set = []
        awbr_set = []
        for category in categories:
            category_both_right = 0
            category_both_wrong = 0
            category_a_right_b_wrong = 0
            category_a_wrong_b_right = 0
            a_detail = run_a[category]["details"]
            b_detail = run_b[category]["details"]
            for qid in a_detail.keys():
                a_acc = next(reversed(a_detail[qid]["acc"].values()))
                a_right = a_acc > 0.5
                b_acc = next(reversed(b_detail[qid]["acc"].values()))
                b_right = b_acc > 0.5
                if a_right and b_right:
                    category_both_right += 1
                elif not a_right and not b_right:
                    category_both_wrong += 1
                elif a_right and not b_right:
                    category_a_right_b_wrong += 1
                    arbw_set.append(b_detail[qid])
                    awbr_set.append(a_detail[qid])
                elif not a_right and b_right:
                    category_a_wrong_b_right += 1
                    awbr_set.append(a_detail[qid])
                    arbw_set.append(b_detail[qid])
            cls._draw_pie(
                dataset_name=dataset_name,
                category=category,
                values=[
                    category_both_right,
                    category_both_wrong,
                    category_a_right_b_wrong,
                    category_a_wrong_b_right,
                ],
                labels=[
                    "Both Right",
                    "Both Wrong",
                    f"{configs[0]['label']} Right, {configs[1]['label']} Wrong",
                    f"{configs[0]['label']} Wrong, {configs[1]['label']} Right",
                ],
                figure_dir=figure_dir,
            )
            both_right += category_both_right
            both_wrong += category_both_wrong
            a_right_b_wrong += category_a_right_b_wrong
            a_wrong_b_right += category_a_wrong_b_right
        cls._draw_pie(
            dataset_name=dataset_name,
            category="all",
            values=[both_right, both_wrong, a_right_b_wrong, a_wrong_b_right],
            labels=[
                "Both Right",
                "Both Wrong",
                f"{configs[0]['label']} Right, {configs[1]['label']} Wrong",
                f"{configs[0]['label']} Wrong, {configs[1]['label']} Right",
            ],
            figure_dir=figure_dir,
        )
        cls._draw_scatter_plot(
            dataset_name=dataset_name,
            category="all",
            data=cls.calculate_scatter_state(
                details=awbr_set,
                competition_type="knockout",
            ),
            competition_type="knockout",
            config=configs[0],
            figure_dir=figure_dir,
            bottom=0.0,
            top=1.0,
            y_label=(r"$\hat{P}_{comp}$"),
        )
        cls._draw_scatter_plot(
            dataset_name=dataset_name,
            category="all",
            data=cls.calculate_scatter_state(
                details=arbw_set,
                competition_type="knockout",
            ),
            competition_type="knockout",
            config=configs[1],
            figure_dir=figure_dir,
            bottom=0.0,
            top=1.0,
            y_label=(r"$\hat{P}_{comp}$"),
        )
