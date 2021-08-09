import json
import warnings
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    COMPARABLE_COLS,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.misc import (
    diff_between_lists,
    find_index_nearest,
    get_lib_alias,
    string_matches_substrings,
)
from sklearn_benchmarks.utils.plotting import make_hover_template, select_front_pareto


def compute_cumulated(fit_times, scores):
    cumulated_fit_times = fit_times.cumsum()
    best_val_score_so_far = pd.Series(scores).cummax()

    return cumulated_fit_times, best_val_score_so_far


def boostrap_fit_times(
    fit_times,
    scores,
    grid_scores,
    n_bootstraps=10_000,
):
    all_fit_times = []
    rng = np.random.RandomState(0)
    n_samples = fit_times.shape[0]
    for _ in range(n_bootstraps):
        indices = rng.randint(n_samples, size=n_samples)
        cum_fit_times_p, cum_scores_p = compute_cumulated(
            fit_times.iloc[indices], scores.iloc[indices]
        )
        grid_fit_times = np.interp(
            grid_scores,
            cum_scores_p,
            cum_fit_times_p,
            right=np.nan,
        )
        all_fit_times.append(grid_fit_times)

    return all_fit_times


@dataclass
class HpoBenchmarkResult:
    """Class to store formatted data of HPO benchmark results."""

    lib: str
    legend: str
    color: str
    df: pd.DataFrame
    fit_times: np.ndarray
    scores: np.ndarray
    mean_grid_times: np.ndarray
    first_quartile_grid_times: np.ndarray
    third_quartile_grid_times: np.ndarray
    grid_scores: np.ndarray


@dataclass
class HpoBenchmarkResults:
    results: List[HpoBenchmarkResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    # Returns base lib (scikit-learn) results
    @property
    def base(self):
        return next(result for result in self.results if result.lib == BASE_LIB)


class HPOReporting:
    def __init__(self, config=None):
        self.config = config

    def set_versions(self):
        with open(VERSIONS_PATH) as json_file:
            self.versions = json.load(json_file)

    def prepare_data(self):
        all_results = []
        estimators = self.config["estimators"]

        baseline_score, max_score = 0.7, 1.0
        grid_scores = np.linspace(baseline_score, max_score, 1000)

        for estimator, params in estimators.items():
            lib = params["lib"]

            df = pd.read_csv(f"{BENCHMARKING_RESULTS_PATH}/{lib}_{estimator}.csv")

            legend = params.get("legend", lib)
            legend += f" ({self.versions[get_lib_alias(lib)]})"

            color = params["color"]

            fit_times = df.query("function == 'fit'")["mean_duration"]
            scores = df.query("function == 'predict' & is_onnx == False")[
                "accuracy_score"
            ]

            bootstrapped_fit_times = boostrap_fit_times(
                fit_times,
                scores,
                grid_scores,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                mean_grid_times = np.nanmean(
                    bootstrapped_fit_times,
                    axis=0,
                )

                first_quartile_grid_times = np.nanpercentile(
                    bootstrapped_fit_times,
                    25,
                    axis=0,
                )

                third_quartile_grid_times = np.nanpercentile(
                    bootstrapped_fit_times,
                    75,
                    axis=0,
                )

            result = HpoBenchmarkResult(
                lib,
                legend,
                color,
                df,
                fit_times,
                scores,
                mean_grid_times,
                first_quartile_grid_times,
                third_quartile_grid_times,
                grid_scores,
            )

            all_results.append(result)

        self.benchmark_results = HpoBenchmarkResults(all_results)

    def scatter(self, func="fit"):
        fig = go.Figure()

        for index, benchmark_result in enumerate(self.benchmark_results):
            df = benchmark_result.df
            df_predictions = df.query("function == 'predict'")

            comparable_cols = COMPARABLE_COLS + ["accuracy_score"]

            comparable_cols = [
                col
                for col in df.columns
                if string_matches_substrings(col, comparable_cols)
            ]
            if benchmark_result.lib == BASE_LIB:
                df_predictions = df_predictions.query("is_onnx == False").merge(
                    df_predictions.query("is_onnx == True")[
                        [
                            "hyperparams_digest",
                            "dataset_digest",
                            *comparable_cols,
                        ]
                    ],
                    on=["hyperparams_digest", "dataset_digest"],
                    how="inner",
                    suffixes=["", "_onnx"],
                )

            df_merged = df_predictions.merge(
                df.query("function == 'fit'")[
                    ["hyperparams_digest", "dataset_digest", *comparable_cols]
                ],
                on=["hyperparams_digest", "dataset_digest"],
                how="inner",
                suffixes=["_predict", "_fit"],
            )

            # Remove columns used during merges
            df_merged = df_merged.drop(
                ["function", "hyperparams_digest", "dataset_digest", "is_onnx"], axis=1
            )

            # Reorder columns for readability purpose in reporting
            ordered_columns = ["n_samples_train", "n_samples", "n_features"]
            for suffix in ["fit", "predict", "onnx"]:
                if benchmark_result.lib != BASE_LIB and suffix == "onnx":
                    continue
                ordered_columns += [
                    f"mean_duration_{suffix}",
                    f"std_duration_{suffix}",
                    f"accuracy_score_{suffix}",
                ]
            ordered_columns = ordered_columns + diff_between_lists(
                df_merged.columns, ordered_columns
            )
            df_merged = df_merged[ordered_columns]
            df_merged = df_merged.dropna(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=df_merged[f"mean_duration_{func}"],
                    y=df_merged[f"accuracy_score_predict"],
                    mode="markers",
                    name=benchmark_result.legend,
                    hovertemplate=make_hover_template(df_merged),
                    customdata=df_merged.values,
                    marker=dict(color=benchmark_result.color),
                    legendgroup=index,
                )
            )

            # Add front pareto line
            data_pareto = df_merged[
                [f"mean_duration_{func}", f"accuracy_score_predict"]
            ]
            df_merged[f"is_pareto_{func}"] = data_pareto.apply(
                select_front_pareto, args=(data_pareto,), axis=1, raw=True
            )
            df_pareto = df_merged.query(f"is_pareto_{func} == True").sort_values(
                [f"mean_duration_{func}"]
            )
            fig.add_trace(
                go.Scatter(
                    x=df_pareto[f"mean_duration_{func}"],
                    y=df_pareto[f"accuracy_score_predict"],
                    mode="lines",
                    showlegend=False,
                    marker=dict(color=benchmark_result.color),
                    legendgroup=index,
                )
            )

            # For scikit-learn, we repeat the process to add ONNX prediction results
            if func == "predict" and benchmark_result.lib == BASE_LIB:
                data_pareto = df_merged[["mean_duration_onnx", "accuracy_score_onnx"]]
                df_merged["is_pareto_onnx"] = data_pareto.apply(
                    select_front_pareto, args=(data_pareto,), axis=1, raw=True
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_merged["mean_duration_onnx"],
                        y=df_merged["accuracy_score_onnx"],
                        mode="markers",
                        name=f"ONNX ({self.versions['onnx']})",
                        hovertemplate=make_hover_template(df_merged),
                        customdata=df_merged.values,
                        marker=dict(color="lightgray"),
                        legendgroup=len(self.benchmark_results),
                    )
                )

                df_pareto = df_merged.query("is_pareto_onnx == True").sort_values(
                    "mean_duration_onnx"
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_pareto["mean_duration_onnx"],
                        y=df_pareto["accuracy_score_onnx"],
                        mode="lines",
                        showlegend=False,
                        marker=dict(color="lightgray"),
                        legendgroup=len(self.benchmark_results),
                    )
                )

        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)

        x_title = f"{func.capitalize()} times in seconds"
        y_title = "Accuracy score"

        fig["layout"]["xaxis{}".format(1)]["title"] = x_title
        fig["layout"]["yaxis{}".format(1)]["title"] = y_title

        fig.show()

    def smoothed_curves(self):
        plt.figure(figsize=(15, 10))

        for benchmark_result in self.benchmark_results:
            plt.plot(
                benchmark_result.mean_grid_times,
                benchmark_result.grid_scores,
                c=f"tab:{benchmark_result.color}",
                label=benchmark_result.legend,
            )

        plt.xlabel("Cumulated fit times in s")
        plt.ylabel("Validation scores")
        plt.legend()
        plt.show()

    def speedup_barplots(self):
        thresholds = self.config["speedup_thresholds"]
        _, axes = plt.subplots(len(thresholds), figsize=(12, 20))

        for ax, threshold in zip(axes, thresholds):
            base_scores = self.benchmark_results.base.grid_scores
            base_fit_times = self.benchmark_results.base.mean_grid_times

            base_idx_closest = find_index_nearest(base_scores, threshold)
            base_time = base_fit_times[base_idx_closest]

            df_threshold = pd.DataFrame(columns=["speedup", "legend", "color"])
            for benchmark_result in self.benchmark_results:
                idx_closest = find_index_nearest(
                    benchmark_result.grid_scores, threshold
                )
                lib_time = benchmark_result.mean_grid_times[idx_closest]
                speedup = base_time / lib_time
                row = dict(
                    speedup=speedup,
                    legend=benchmark_result.legend,
                    color=benchmark_result.color,
                )
                df_threshold = df_threshold.append(row, ignore_index=True)

            ax.bar(
                x=df_threshold["legend"],
                height=df_threshold["speedup"],
                width=0.3,
                color=df_threshold["color"],
            )
            ax.set_xlabel("Library")
            ax.set_ylabel(f"Speedup")
            ax.set_title(f"At score {threshold}")

        plt.tight_layout()
        plt.show()

    def speedup_curves(self):
        plt.figure(figsize=(15, 10))
        for benchmark_result in self.benchmark_results:
            plt.plot(
                benchmark_result.grid_scores,
                self.benchmark_results.base.mean_grid_times
                / benchmark_result.mean_grid_times,
                c=f"tab:{benchmark_result.color}",
                label=benchmark_result.legend,
            )
            plt.fill_between(
                benchmark_result.grid_scores,
                self.benchmark_results.base.third_quartile_grid_times
                / benchmark_result.third_quartile_grid_times,
                self.benchmark_results.base.first_quartile_grid_times
                / benchmark_result.first_quartile_grid_times,
                color=benchmark_result.color,
                alpha=0.1,
            )
        plt.xlabel("Validation scores")
        plt.ylabel(f"Speedup")
        plt.legend()
        plt.show()

    def make_report(self):
        config = get_full_config(config=self.config)
        self.config = config["hpo_reporting"]

        self.set_versions()

        self.prepare_data()

        display(Markdown("## Benchmark results for `fit`"))
        display(Markdown("### Raw fit times vs. accuracy scores"))
        self.scatter(func="fit")

        display(Markdown("### Smoothed HPO Curves"))
        display(Markdown("> The shaded areas represent boostrapped quartiles."))
        self.smoothed_curves()

        display(Markdown("### Speedup barplots"))
        self.speedup_barplots()

        display(Markdown("### Speedup curves"))
        self.speedup_curves()

        display(Markdown("## Benchmark results for `predict`"))
        display(Markdown("### Raw predict times vs. accuracy scores"))
        self.scatter(func="predict")
