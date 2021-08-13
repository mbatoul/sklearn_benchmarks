import importlib
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots
from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    COMPARABLE_COLS,
    PLOT_HEIGHT_IN_PX,
    REPORTING_FONT_SIZE,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils import (
    diff_between_lists,
    gen_coordinates_grid,
    get_lib_alias,
    make_hover_template,
)


class HpMatchReporting:
    """
    Runs reporting for specified estimators.
    """

    def __init__(self, against_lib="", config=None):
        self.against_lib = against_lib
        self.config = config

    def _get_estimator_default_parameters(self, estimator):
        splitted_path = estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        estimator_class = getattr(importlib.import_module(module), class_name)
        estimator_instance = estimator_class()
        parameters = estimator_instance.__dict__.keys()
        return parameters

    def _get_estimator_parameters(self, estimator_config):
        if "parameters" in estimator_config:
            return estimator_config["parameters"]["init"].keys()
        else:
            return self._get_estimator_default_parameters(estimator_config["estimator"])

    def make_report(self):
        config = get_full_config(config=self.config)
        reporting_config = config["hp_match_reporting"]
        benchmarking_estimators = config["benchmarking"]["estimators"]
        reporting_estimators = reporting_config["estimators"][self.against_lib]

        with open(VERSIONS_PATH) as json_file:
            versions = json.load(json_file)

        description = (
            "We assume here there is a perfect match between the hyperparameters of both librairies. "
            f"For a given set of parameters and a given dataset, we compute the speedup `time scikit-learn / time {self.against_lib}`. "
            f"For instance, a speedup of 2 means that {self.against_lib} is twice as fast as scikit-learn for a given set of parameters and a given dataset."
        )
        display(Markdown(f"> {description}"))

        for name, params in reporting_estimators.items():
            params["n_cols"] = reporting_config["n_cols"]
            params["estimator_parameters"] = self._get_estimator_parameters(
                benchmarking_estimators[name]
            )
            against_lib = params["against_lib"]
            against_lib = get_lib_alias(against_lib)
            title = f"## `{name}`"
            subtitle = f"**{against_lib} ({versions[against_lib]}) vs. scikit-learn ({versions['scikit-learn']})**"
            display(Markdown(title))
            display(Markdown(subtitle))

            report = SingleEstimatorReporting(**params)
            report.make_report()


def make_profiling_link(components, lib=BASE_LIB):
    function, parameters_digest, dataset_digest = components
    path = f"profiling/{lib}_{function}_{parameters_digest}_{dataset_digest}.html"

    if os.environ.get("RESULTS_BASE_URL") is not None:
        base_url = os.environ.get("RESULTS_BASE_URL")
    else:
        base_url = "http://localhost:8000/results/"

    return f"<a href='{base_url}{path}' target='_blank'>See</a>"


def add_bar_plotly(
    fig,
    row,
    col,
    df,
    color="dodgerblue",
    name="",
    showlegend=False,
):
    x = [f"({ns}, {nf})" for ns, nf in df[["n_samples", "n_features"]].values]
    y = df["speedup"]

    bar = go.Bar(
        x=x,
        y=y,
        name=name,
        marker_color=color,
        hovertemplate=make_hover_template(df),
        customdata=df.values,
        showlegend=showlegend,
        text=df["function"],
        textposition="auto",
    )
    fig.add_trace(
        bar,
        row=row,
        col=col,
    )


class SingleEstimatorReporting:
    """
    Runs reporting for one estimator.
    """

    def __init__(
        self,
        name="",
        against_lib="",
        split_bars_by=[],
        estimator_parameters={},
        n_cols=None,
    ):
        self.name = name
        self.against_lib = against_lib
        self.split_bars_by = split_bars_by
        self.n_cols = n_cols
        self.estimator_parameters = estimator_parameters

    def _get_benchmark_df(self, lib=BASE_LIB):
        benchmarking_results_path = str(BENCHMARKING_RESULTS_PATH)
        file_path = f"{benchmarking_results_path}/{lib}_{self.name}.csv"

        return pd.read_csv(file_path)

    def prepare_data(self):
        base_lib_df = self._get_benchmark_df()
        base_lib_time = base_lib_df["mean_duration"]
        base_lib_std = base_lib_df["mean_duration"]

        against_lib_df = self._get_benchmark_df(lib=self.against_lib)
        against_lib_time = against_lib_df["mean_duration"]
        against_lib_std = against_lib_df["std_duration"]

        comparable_cols = [
            col for col in COMPARABLE_COLS if col in against_lib_df.columns
        ]

        suffixes = map(lambda lib: f"_{lib}", [BASE_LIB, self.against_lib])

        df_reporting = pd.merge(
            base_lib_df,
            against_lib_df[comparable_cols],
            left_index=True,
            right_index=True,
            suffixes=suffixes,
        )

        df_reporting["speedup"] = base_lib_time / against_lib_time
        df_reporting["std_speedup"] = df_reporting["speedup"] * (
            np.sqrt(
                (base_lib_std / base_lib_time) ** 2
                + (against_lib_std / against_lib_time) ** 2
            )
        )

        self.df_reporting = df_reporting

    def make_plot_title(self, df):
        title = ""
        params_cols = [
            param
            for param in self.estimator_parameters
            if param not in self.split_bars_by
            and param not in self.get_shared_parameters().keys()
        ]
        values = df[params_cols].values[0]

        for index, (param, value) in enumerate(zip(params_cols, values)):
            title += "%s: %s" % (param, value)
            if index != len(list(enumerate(zip(params_cols, values)))) - 1:
                title += "<br>"

        return title

    def print_tables(self):
        df = self.df_reporting

        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        cols_to_drop = [col for col in cols_to_drop if col in self.estimator_parameters]

        df = df.drop(cols_to_drop, axis=1)
        df = df.dropna(axis=1)
        df = df.round(3)

        for lib in [BASE_LIB, self.against_lib]:
            df[f"{lib}_profiling"] = df[
                ["function", "parameters_digest", "dataset_digest"]
            ].apply(make_profiling_link, lib=lib, axis=1)

        df = df.drop(["parameters_digest", "dataset_digest"], axis=1)
        dfs = [x for _, x in df.groupby(["function"])]

        for df in dfs:
            display(HTML(df.to_html(escape=False)))

    def get_shared_parameters(self):
        df = self.df_reporting

        shared_params = {}
        for col in self.estimator_parameters:
            unique_vals = df[col].unique()
            if unique_vals.size == 1:
                shared_params[col] = unique_vals[0]

        return shared_params

    def plot(self):
        df_reporting = self.df_reporting

        comparable_cols = [
            col for col in COMPARABLE_COLS if col in df_reporting.columns
        ]

        # Reorder columns for readability purpose in reporting
        ordered_columns = [
            "estimator",
            "n_samples_train",
            "n_samples",
            "n_features",
        ]

        for col in comparable_cols:
            for suffix in ["fit", "predict", "onnx"]:
                ordered_columns += [f"{col}_{suffix}"]

        ordered_columns = ordered_columns + diff_between_lists(
            df_reporting.columns, ordered_columns
        )

        if self.split_bars_by:
            group_by_params = [
                param
                for param in self.estimator_parameters
                if param not in self.split_bars_by
            ]
        else:
            group_by_params = ["parameters_digest"]

        df_reporting_grouped = df_reporting.groupby([*group_by_params, "function"])

        n_plots = len(df_reporting_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = gen_coordinates_grid(n_rows, self.n_cols)

        subplot_titles = [self.make_plot_title(df) for _, df in df_reporting_grouped]

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

        for (row, col), (_, df) in zip(coordinates, df_reporting_grouped):
            df = df.sort_values(by=["function", "n_samples", "n_features"])
            df = df.dropna(axis="columns")
            df = df.drop(["parameters_digest", "dataset_digest"], axis=1)
            df = df.round(3)

            if self.split_bars_by:
                for split_col in self.split_bars_by:
                    split_col_vals = df[split_col].unique()
                    for index, split_val in enumerate(split_col_vals):
                        df_filtered = df[df[split_col] == split_val]
                        df_filtered = df_filtered.sort_values(
                            by=["function", "n_samples", "n_features"]
                        )
                        add_bar_plotly(
                            fig,
                            row,
                            col,
                            df_filtered,
                            color=px.colors.qualitative.Plotly[index],
                            name="%s: %s" % (split_col, split_val),
                            showlegend=(row, col) == (1, 1),
                        )
            else:
                add_bar_plotly(fig, row, col, df)

        for i in range(1, n_plots + 1):
            fig["layout"]["xaxis{}".format(i)]["title"] = "(n_samples, n_features)"
            fig["layout"]["yaxis{}".format(i)]["title"] = "Speedup in logarithmic scale"

        fig.for_each_xaxis(
            lambda axis: axis.title.update(font=dict(size=REPORTING_FONT_SIZE))
        )
        fig.for_each_yaxis(
            lambda axis: axis.title.update(font=dict(size=REPORTING_FONT_SIZE))
        )
        fig.update_annotations(font_size=REPORTING_FONT_SIZE)
        fig.update_layout(
            height=n_rows * PLOT_HEIGHT_IN_PX, barmode="group", showlegend=True
        )
        for row, col in coordinates:
            fig.update_yaxes(type="log", row=row, col=col)

        text = "All estimators share the following parameters: "
        df_shared_parameters = pd.DataFrame.from_dict(
            self.get_shared_parameters(), orient="index", columns=["value"]
        )
        for i, (index, row) in enumerate(df_shared_parameters.iterrows()):
            text += "`%s=%s`" % (index, *row.values)
            if i == len(df_shared_parameters) - 1:
                text += "."
            else:
                text += ", "
        display(Markdown(text))

        fig.show()

    def check_scores_are_close(self):
        df = self.prepare_data()

        df_filtered = df.copy()
        scores = [col for col in df_filtered.columns if "score" in col]
        scores = set(list(map(lambda score: "_".join(score.split("_")[:-1]), scores)))

        for score in scores:

            df_filtered[f"diff_{score}s"] = np.absolute(
                df_filtered[f"{score}_{BASE_LIB}"]
                - df_filtered[f"{score}_{self.against_lib}"]
            )
            df_filtered = df_filtered.query("function == 'predict'")

            threshold = 0.001
            df_filtered = df_filtered.query(f"diff_{score}s >= {threshold}")

        if not df_filtered.empty:
            n_mismatches = len(df_filtered)
            n_total_predictions = len(df.query("function == 'predict'"))

            proportion_mismatch = n_mismatches / n_total_predictions * 100
            proportion_mismatch = round(proportion_mismatch, 2)

            word_prediction = "prediction"
            if n_mismatches > 1:
                word_prediction += "s"

            display(
                HTML(
                    "<div style='padding: 20px; background-color: #f44336; color: white; margin-bottom: 15px;'>"
                    "<strong>WARNING!</strong> "
                    f"Mismatch between validation scores for {n_mismatches} {word_prediction} ({proportion_mismatch}%). See details in the dataframe below."
                    "</div>"
                )
            )

            relevant_cols = [
                "estimator",
                "function",
                "n_samples_train",
                "n_samples",
                "n_features",
            ]

            for score in scores:
                relevant_cols += [
                    f"{score}_{BASE_LIB}",
                    f"{score}_{self.against_lib}",
                    f"diff_{score}s",
                ]

            display(df_filtered[relevant_cols])

    def make_report(self):
        self.prepare_data()

        display(Markdown("### Speedup barplots"))
        self.plot()

        self.check_scores_are_close()

        display(Markdown("### Raw results"))
        self.print_tables()
