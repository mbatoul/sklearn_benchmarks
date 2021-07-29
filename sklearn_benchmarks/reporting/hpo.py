import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    HPO_CURVES_COLORS,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.misc import find_nearest
from sklearn_benchmarks.utils.plotting import (
    identify_pareto,
    make_hover_template,
    mean_bootstrapped_curve,
    order_columns,
    quartile_bootstrapped_curve,
)


class HPOReporting:
    def __init__(self, config=None):
        self.config = config

    def _set_versions(self):
        with open(VERSIONS_PATH) as json_file:
            self._versions = json.load(json_file)

    def _display_scatter(self, func="fit"):
        fig = go.Figure()

        for index, params in enumerate(self._config["estimators"]):
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)
            df = df.fillna(value={"use_onnx_runtime": False})

            legend = params.get("lib")
            legend = params.get("legend", legend)
            key_lib_version = params["lib"]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            legend += f" ({self._versions[key_lib_version]})"

            if "use_onnx_runtime" in df.columns:
                df = df.query("use_onnx_runtime == False")

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

            color = HPO_CURVES_COLORS[index]

            df_hover = df_merged.copy()
            df_hover.columns = df_hover.columns.str.replace(f"_{func}", "")
            df_hover = df_hover.rename(
                columns={"mean": f"mean_{func}_time", "stdev": f"stdev_{func}_time"}
            )
            df_hover = df_hover[
                df_hover.columns.drop(list(df_hover.filter(regex="digest")))
            ]
            df_hover = df_hover.round(3)

            fig.add_trace(
                go.Scatter(
                    x=df_merged[f"mean_{func}"],
                    y=df_merged["accuracy_score"],
                    mode="markers",
                    name=legend,
                    hovertemplate=make_hover_template(df_hover),
                    customdata=df_hover[order_columns(df_hover)].values,
                    marker=dict(color=color),
                    legendgroup=index,
                )
            )

            data = df_merged[[f"mean_{func}", "accuracy_score"]].values
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

            if "use_onnx_runtime" in df.columns and func == "predict":
                # Add ONNX points
                legend = f"ONNX ({self._versions['onnx']})"
                color = HPO_CURVES_COLORS[len(self._config["estimators"])]

                df = pd.read_csv(file)
                df = df.fillna(value={"use_onnx_runtime": False})
                df_merged = df.query("function == 'predict' & use_onnx_runtime == True")

                df_hover = df_merged.copy()
                df_hover = df_hover[
                    df_hover.columns.drop(list(df_hover.filter(regex="digest")))
                ]
                df_hover = df_hover.round(3)

                fig.add_trace(
                    go.Scatter(
                        x=df_merged[f"mean"],
                        y=df_merged["accuracy_score"],
                        mode="markers",
                        name=legend,
                        hovertemplate=make_hover_template(df_hover),
                        customdata=df_hover[order_columns(df_hover)].values,
                        marker=dict(color=color),
                        legendgroup=len(self._config["estimators"]),
                    )
                )

                data = df_merged[[f"mean", "accuracy_score"]].values
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

    def display_smoothed_curves(self):
        plt.figure(figsize=(15, 10))

        fit_times_for_max_scores = []
        for index, params in enumerate(self._config["estimators"]):
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)

            legend = params.get("lib")
            legend = params.get("legend", legend)

            key_lib_version = params["lib"]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            legend += f" ({self._versions[key_lib_version]})"

            fit_times = df[df["function"] == "fit"]["mean"]
            scores = df[df["function"] == "predict"]["accuracy_score"]

            color = HPO_CURVES_COLORS[index]

            mean_grid_times, grid_scores = mean_bootstrapped_curve(fit_times, scores)
            idx_max_score = np.argmax(grid_scores, axis=0)
            fit_time_for_max_score = mean_grid_times[idx_max_score]
            fit_times_for_max_scores.append(fit_time_for_max_score)

            plt.plot(mean_grid_times, grid_scores, c=f"tab:{color}", label=legend)

        min_fit_time_all_constant = min(fit_times_for_max_scores)
        plt.xlim(right=min_fit_time_all_constant)
        plt.xlabel("Cumulated fit times in s")
        plt.ylabel("Validation scores")
        plt.legend()
        plt.show()

    def display_speedup_barplots(self):
        other_lib_dfs = {}
        for params in self._config["estimators"]:
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)

            if params["lib"] == BASE_LIB:
                base_lib_df = df
                base_lib_df = base_lib_df.fillna(value={"use_onnx_runtime": False})
                base_lib_df = base_lib_df.query("use_onnx_runtime == False")

            key = "".join(params.get("legend", params.get("lib")))
            other_lib_dfs[key] = df

        data = []
        columns = ["score"]

        base_fit_times = base_lib_df[base_lib_df["function"] == "fit"]["mean"]
        base_scores = base_lib_df[base_lib_df["function"] == "predict"][
            "accuracy_score"
        ]

        for val in self._config["speedup"]["scores"]:
            row = [val]
            for lib, df in other_lib_dfs.items():
                if lib not in columns:
                    columns.append(lib)

                if "use_onnx_runtime" in df.columns:
                    df = df.fillna(value={"use_onnx_runtime": False})
                    df = df.query("use_onnx_runtime == False")

                fit_times = df[df["function"] == "fit"]["mean"]
                scores = df[df["function"] == "predict"]["accuracy_score"]

                assert fit_times.shape == scores.shape

                idx, _ = find_nearest(scores, val)
                other_lib_time = fit_times.iloc[idx]

                idx, _ = find_nearest(base_scores, val)
                base_time = base_fit_times.iloc[idx]

                speedup = base_time / other_lib_time

                row.append(speedup)
            data.append(row)

        speedup_df = pd.DataFrame(columns=columns, data=data)
        speedup_df = speedup_df.set_index("score")
        _, axes = plt.subplots(3, figsize=(12, 20))

        libs = list(speedup_df.columns)
        for i in range(len(libs)):
            key_lib_version = libs[i].split(" ")[0]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            libs[i] = f"{libs[i]} ({self._versions[key_lib_version]})"

        for ax, score in zip(axes, speedup_df.index.unique()):
            speedups = speedup_df.loc[score].values
            ax.bar(x=libs, height=speedups, width=0.3, color=HPO_CURVES_COLORS)
            ax.set_xlabel("Lib")
            ax.set_ylabel(f"Speedup (time sklearn / time lib)")
            ax.set_title(f"At score {score}")

        plt.tight_layout()
        plt.show()

    def display_speedup_curves(self):
        other_lib_dfs = {}
        for params in self._config["estimators"]:
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)
            if params["lib"] == BASE_LIB:
                base_lib_df = df
            else:
                key = "".join(params.get("legend", params.get("lib")))
                other_lib_dfs[key] = df

        base_fit_times = base_lib_df[base_lib_df["function"] == "fit"]["mean"]
        base_scores = base_lib_df[base_lib_df["function"] == "predict"][
            "accuracy_score"
        ]
        base_mean_grid_times, base_grid_scores = mean_bootstrapped_curve(
            base_fit_times, base_scores
        )
        base_first_quartile_fit_times, _ = quartile_bootstrapped_curve(
            base_fit_times, base_scores, 25
        )
        base_third_quartile_fit_times, _ = quartile_bootstrapped_curve(
            base_fit_times, base_scores, 75
        )
        plt.figure(figsize=(15, 10))

        base_lib_alias = self._config["version_aliases"][BASE_LIB]
        label = f"{base_lib_alias} ({self._versions[base_lib_alias]})"
        plt.plot(
            base_grid_scores,
            base_mean_grid_times / base_mean_grid_times,
            c=HPO_CURVES_COLORS[0],
            label=label,
        )

        for index, (lib, df) in enumerate(other_lib_dfs.items()):
            fit_times = df[df["function"] == "fit"]["mean"]
            scores = df[df["function"] == "predict"]["accuracy_score"]

            mean_grid_times, grid_scores = mean_bootstrapped_curve(fit_times, scores)
            speedup_mean = base_mean_grid_times / mean_grid_times

            color = HPO_CURVES_COLORS[index + 1]

            key_lib_version = lib.split(" ")[0]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            label = f"{lib} ({self._versions[key_lib_version]})"

            plt.plot(grid_scores, speedup_mean, c=f"tab:{color}", label=label)

            first_quartile_grid_times, _ = quartile_bootstrapped_curve(
                fit_times, scores, 25
            )
            speedup_first_quartile = (
                base_first_quartile_fit_times / first_quartile_grid_times
            )

            third_quartile_grid_times, _ = quartile_bootstrapped_curve(
                fit_times, scores, 75
            )
            speedup_third_quartile = (
                base_third_quartile_fit_times / third_quartile_grid_times
            )

            plt.fill_between(
                grid_scores,
                speedup_third_quartile,
                speedup_first_quartile,
                color=color,
                alpha=0.1,
            )

        plt.xlabel("Validation scores")
        plt.ylabel(f"Speedup vs. {BASE_LIB}")
        plt.legend()
        plt.show()

    def run(self):
        config = get_full_config(config=self.config)
        self._config = config["hpo_reporting"]

        self._set_versions()

        display(Markdown("## Raw fit times vs. accuracy scores"))
        self._display_scatter(func="fit")

        display(Markdown("## Raw predict times vs. accuracy scores"))
        self._display_scatter(func="predict")

        display(Markdown("## Smoothed HPO Curves"))
        display(Markdown("> The shaded areas represent boostrapped quartiles."))
        self.display_smoothed_curves()

        display(Markdown("## Speedup barplots"))
        self.display_speedup_barplots()

        display(Markdown("## Speedup curves"))
        self.display_speedup_curves()
