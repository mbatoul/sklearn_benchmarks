import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.misc import find_nearest, get_lib_alias
from sklearn_benchmarks.utils.plotting import (
    identify_pareto,
    make_hover_template,
    mean_bootstrapped_curve,
    order_columns,
    percentile_bootstrapped_curve,
)


@dataclass
class BenchResult:
    """Class to store formatted data of HPO benchmark results."""

    lib: str
    legend: str
    color: str
    df: pd.DataFrame
    fit_times: np.ndarray
    scores: np.ndarray
    mean_bootstrapped_grid_times: np.ndarray
    first_quartile_bootstrapped_grid_times: np.ndarray
    third_quartile_bootstrapped_grid_times: np.ndarray
    grid_scores: np.ndarray


@dataclass
class BenchResults:
    results: list[BenchResult]

    def __iter__(self):
        return iter(self.results)

    @property
    def base(self):
        return next(result for result in self.results if result.lib == BASE_LIB)


class HPOReporting:
    def __init__(self, config=None):
        self.config = config

    def _set_versions(self):
        with open(VERSIONS_PATH) as json_file:
            self._versions = json.load(json_file)

    def _prepare_data(self):
        all_data = []
        estimators = self.config["estimators"]

        for estimator, params in estimators.items():
            lib = params["lib"]

            df = pd.read_csv(f"{BENCHMARKING_RESULTS_PATH}/{lib}_{estimator}.csv")

            legend = params.get("legend", lib)
            legend += f" ({self._versions[get_lib_alias(lib)]})"

            color = params["color"]

            fit_times = df.query("function == 'fit'")["mean_duration"]
            scores = df.query("function == 'predict' & is_onnx == False")[
                "accuracy_score"
            ]
            mean_bootstrapped_grid_times, grid_scores = mean_bootstrapped_curve(
                fit_times, scores
            )
            (
                first_quartile_bootstrapped_grid_times,
                _,
            ) = percentile_bootstrapped_curve(fit_times, scores, 25)
            (
                third_quartile_bootstrapped_grid_times,
                _,
            ) = percentile_bootstrapped_curve(fit_times, scores, 75)

            result = BenchResult(
                lib,
                legend,
                color,
                df,
                fit_times,
                scores,
                mean_bootstrapped_grid_times,
                first_quartile_bootstrapped_grid_times,
                third_quartile_bootstrapped_grid_times,
                grid_scores,
            )

            all_data.append(result)

            contains_onnx_predictions = df[["is_onnx"]].any(axis=None, bool_only=True)
            if contains_onnx_predictions:
                lib = "onnx"
                legend = f"ONNX ({self._versions['onnx']})"
                color = "lightgray"
                scores = df.query("function == 'predict' & is_onnx == True")[
                    "accuracy_score"
                ]
                mean_bootstrapped_grid_times, grid_scores = mean_bootstrapped_curve(
                    fit_times, scores
                )
                df["accuracy_score"] = df.query("is_onnx == True")["accuracy_score"]

                result_onnx = BenchResult(
                    lib,
                    legend,
                    color,
                    df,
                    fit_times,
                    scores,
                    mean_bootstrapped_grid_times,
                    grid_scores,
                )

                all_data.append(result_onnx)

        self.bench_results = BenchResults(all_data)

    def scatter(self, func="fit"):
        fig = go.Figure()

        for index, (name, params) in enumerate(self.config["estimators"].items()):
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{name}.csv"
            df = pd.read_csv(file)
            df = df.fillna(value={"is_onnx": False})

            lib = params.get("lib")
            legend = params.get("legend", lib)
            lib = get_lib_alias(lib)
            legend += f" ({self._versions[lib]})"

            if "is_onnx" in df.columns:
                df = df.query("is_onnx == False")

            df_merged = df.query("function == 'fit'").merge(
                df.query("function == 'predict'"),
                on=["hyperparams_digest", "dataset_digest"],
                how="inner",
                suffixes=["_fit", "_predict"],
            )
            df_merged = df_merged.dropna(axis=1)
            suffix_to_drop = "_predict" if func == "fit" else "_fit"
            df_merged = df_merged.rename(
                columns={"accuracy_score_predict": "accuracy_score"}
            )
            df_merged.drop(
                df_merged.filter(regex=f"{suffix_to_drop}$").columns.tolist(),
                axis=1,
                inplace=True,
            )

            color = params["color"]

            df_hover = df_merged.copy()
            df_hover.columns = df_hover.columns.str.replace(f"_{func}", "")
            df_hover = df_hover.rename(
                columns={
                    "mean_duration": f"mean_duration_{func}",
                    "std_duration": f"std_duration_{func}",
                }
            )
            df_hover = df_hover[
                df_hover.columns.drop(list(df_hover.filter(regex="digest")))
            ]
            df_hover = df_hover.round(3)

            fig.add_trace(
                go.Scatter(
                    x=df_merged[f"mean_duration_{func}"],
                    y=df_merged["accuracy_score"],
                    mode="markers",
                    name=legend,
                    hovertemplate=make_hover_template(df_hover),
                    customdata=df_hover[order_columns(df_hover)].values,
                    marker=dict(color=color),
                    legendgroup=index,
                )
            )

            data = df_merged[[f"mean_duration_{func}", "accuracy_score"]].values
            pareto_indices = identify_pareto(data)
            pareto_front = data[pareto_indices]
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            fig.add_trace(
                go.Scatter(
                    x=pareto_front[:, 0],
                    y=pareto_front[:, 1],
                    mode="lines",
                    showlegend=False,
                    marker=dict(color=color),
                    legendgroup=index,
                )
            )

            if "is_onnx" in df.columns and func == "predict":
                # Add ONNX points
                legend = f"ONNX ({self._versions['onnx']})"
                color = "lightgray"

                df = pd.read_csv(file)
                df = df.fillna(value={"is_onnx": False})
                df_merged = df.query("function == 'predict' & is_onnx == True")

                df_hover = df_merged.copy()
                df_hover = df_hover[
                    df_hover.columns.drop(list(df_hover.filter(regex="digest")))
                ]
                df_hover = df_hover.round(3)

                fig.add_trace(
                    go.Scatter(
                        x=df_merged[f"mean_duration"],
                        y=df_merged["accuracy_score"],
                        mode="markers",
                        name=legend,
                        hovertemplate=make_hover_template(df_hover),
                        customdata=df_hover[order_columns(df_hover)].values,
                        marker=dict(color=color),
                        legendgroup=len(self.config["estimators"]),
                    )
                )

                data = df_merged[[f"mean_duration", "accuracy_score"]].values
                pareto_indices = identify_pareto(data)
                pareto_front = data[pareto_indices]
                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values

                fig.add_trace(
                    go.Scatter(
                        x=pareto_front[:, 0],
                        y=pareto_front[:, 1],
                        mode="lines",
                        showlegend=False,
                        marker=dict(color=color),
                        legendgroup=index,
                    )
                )

        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        fig["layout"]["xaxis{}".format(1)][
            "title"
        ] = f"{func.capitalize()} times in seconds"
        fig["layout"]["yaxis{}".format(1)]["title"] = "Accuracy score"
        fig.show()

    def smoothed_curves(self):
        plt.figure(figsize=(15, 10))

        fit_times_for_max_scores = []

        for bench_result in self.bench_results:
            idx_max_score = np.argmax(bench_result.grid_scores, axis=0)
            fit_time_for_max_score = bench_result.mean_bootstrapped_grid_times[
                idx_max_score
            ]
            fit_times_for_max_scores.append(fit_time_for_max_score)
            plt.plot(
                bench_result.mean_bootstrapped_grid_times,
                bench_result.grid_scores,
                c=f"tab:{bench_result.color}",
                label=bench_result.legend,
            )

        min_fit_time_all_constant = min(fit_times_for_max_scores)
        plt.xlim(right=min_fit_time_all_constant)
        plt.xlabel("Cumulated fit times in s")
        plt.ylabel("Validation scores")
        plt.legend()
        plt.show()

    def speedup_barplots(self):
        thresholds = self.config["speedup_thresholds"]
        _, axes = plt.subplots(len(thresholds), figsize=(12, 20))

        base_bench_result = list(
            filter(lambda result: result.lib == BASE_LIB, self.bench_results)
        )[0]

        for ax, threshold in zip(axes, thresholds):
            base_scores = base_bench_result.grid_scores
            base_fit_times = base_bench_result.mean_bootstrapped_grid_times

            base_idx_closest, _ = find_nearest(base_scores, threshold)
            base_time = base_fit_times[base_idx_closest]

            df_threshold = pd.DataFrame(columns=["speedup", "legend", "color"])
            for bench_result in self.bench_results:
                idx_closest, _ = find_nearest(bench_result.grid_scores, threshold)
                lib_time = bench_result.mean_bootstrapped_grid_times[idx_closest]
                speedup = base_time / lib_time
                row = dict(
                    speedup=speedup,
                    legend=bench_result.legend,
                    color=bench_result.color,
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
        for bench_result in self.bench_results:
            plt.plot(
                bench_result.grid_scores,
                self.bench_results.base.mean_bootstrapped_grid_times
                / bench_result.mean_bootstrapped_grid_times,
                c=f"tab:{bench_result.color}",
                label=bench_result.legend,
            )
            plt.fill_between(
                bench_result.grid_scores,
                self.bench_results.base.third_quartile_bootstrapped_grid_times
                / bench_result.third_quartile_bootstrapped_grid_times,
                self.bench_results.base.first_quartile_bootstrapped_grid_times
                / bench_result.first_quartile_bootstrapped_grid_times,
                color=bench_result.color,
                alpha=0.1,
            )
        plt.xlabel("Validation scores")
        plt.ylabel(f"Speedup")
        plt.legend()
        plt.show()

    def run(self):
        config = get_full_config(config=self.config)
        self.config = config["hpo_reporting"]

        self._set_versions()

        self._prepare_data()

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
