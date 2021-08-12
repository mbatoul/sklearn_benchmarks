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
    gen_coordinates_grid,
    get_lib_alias,
    make_hover_template,
)


class HPMatchReporting:
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

            report = SingleEstimatorReport(**params)
            report.make_report()


class SingleEstimatorReport:
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

    def _make_reporting_df(self):
        base_lib_df = self._get_benchmark_df()
        base_lib_time = base_lib_df["mean_duration"]
        base_lib_std = base_lib_df["mean_duration"]

        against_lib_df = self._get_benchmark_df(lib=self.against_lib)
        against_lib_time = against_lib_df["mean_duration"]
        against_lib_std = against_lib_df["std_duration"]

        suffixes = map(lambda lib: f"_{lib}", [BASE_LIB, self.against_lib])
        df_merged = pd.merge(
            base_lib_df,
            against_lib_df[COMPARABLE_COLS],
            left_index=True,
            right_index=True,
            suffixes=suffixes,
        )

        df_merged["speedup"] = base_lib_time / against_lib_time
        df_merged["std_speedup"] = df_merged["speedup"] * (
            np.sqrt(
                (base_lib_std / base_lib_time) ** 2
                + (against_lib_std / against_lib_time) ** 2
            )
        )

        return df_merged

    def _make_profiling_link(self, components, lib=BASE_LIB):
        function, parameters_digest, dataset_digest = components
        path = f"profiling/{lib}_{function}_{parameters_digest}_{dataset_digest}.html"
        if os.environ.get("RESULTS_BASE_URL") is not None:
            base_url = os.environ.get("RESULTS_BASE_URL")
        else:
            base_url = "http://localhost:8000/results/"
        return f"<a href='{base_url}{path}' target='_blank'>See</a>"

    def _make_plot_title(self, df):
        title = ""
        params_cols = [
            param
            for param in self.estimator_parameters
            if param not in self.split_bars_by
            and param not in self._get_shared_hyperpameters().keys()
        ]
        values = df[params_cols].values[0]
        for index, (param, value) in enumerate(zip(params_cols, values)):
            title += "%s: %s" % (param, value)
            if index != len(list(enumerate(zip(params_cols, values)))) - 1:
                title += "<br>"
        return title

    def _print_tables(self):
        df = self._make_reporting_df()
        df = df.round(3)
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        cols_to_drop = [col for col in cols_to_drop if col in self.estimator_parameters]
        df = df.drop(cols_to_drop, axis=1)
        for lib in [BASE_LIB, self.against_lib]:
            df[f"{lib}_profiling"] = df[
                ["function", "parameters_digest", "dataset_digest"]
            ].apply(self._make_profiling_link, lib=lib, axis=1)
        df = df.drop(["parameters_digest", "dataset_digest"], axis=1)
        splitted_dfs = [x for _, x in df.groupby(["function"])]
        for df in splitted_dfs:
            display(HTML(df.to_html(escape=False)))

    def _make_x_plot(self, df):
        return [f"({ns}, {nf})" for ns, nf in df[["n_samples", "n_features"]].values]

    def _add_bar_to_plotly_fig(
        self, fig, row, col, df, color="dodgerblue", name="", showlegend=False
    ):
        bar = go.Bar(
            x=self._make_x_plot(df),
            y=df["speedup"],
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

    def _get_shared_hyperpameters(self):
        df_merged = self._make_reporting_df()
        ret = {}
        for col in self.estimator_parameters:
            unique_vals = df_merged[col].unique()
            if unique_vals.size == 1:
                ret[col] = unique_vals[0]
        return ret

    def _plot(self):
        df_merged = self._make_reporting_df()

        if self.split_bars_by:
            group_by_params = [
                param
                for param in self.estimator_parameters
                if param not in self.split_bars_by
            ]
        else:
            group_by_params = "parameters_digest"

        df_merged_grouped = df_merged.groupby(group_by_params)

        n_plots = len(df_merged_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = gen_coordinates_grid(n_rows, self.n_cols)

        subplot_titles = [self._make_plot_title(df) for _, df in df_merged_grouped]

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

        for (row, col), (_, df) in zip(coordinates, df_merged_grouped):
            df = df.sort_values(by=["function", "n_samples", "n_features"])
            df = df.dropna(axis="columns")
            df = df.drop(["parameters_digest", "dataset_digest"], axis=1)
            df = df.round(3)

            if self.split_bars_by:
                for split_col in self.split_bars_by:
                    split_col_vals = df[split_col].unique()
                    for index, split_val in enumerate(split_col_vals):
                        filtered_df = df[df[split_col] == split_val]
                        filtered_df = filtered_df.sort_values(
                            by=["function", "n_samples", "n_features"]
                        )
                        self._add_bar_to_plotly_fig(
                            fig,
                            row,
                            col,
                            filtered_df,
                            color=px.colors.qualitative.Plotly[index],
                            name="%s: %s" % (split_col, split_val),
                            showlegend=(row, col) == (1, 1),
                        )
            else:
                self._add_bar_to_plotly_fig(fig, row, col, df)

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
            self._get_shared_hyperpameters(), orient="index", columns=["value"]
        )
        for i, (index, row) in enumerate(df_shared_parameters.iterrows()):
            text += "`%s=%s`" % (index, *row.values)
            if i == len(df_shared_parameters) - 1:
                text += "."
            else:
                text += ", "
        display(Markdown(text))

        fig.show()

    def make_report(self):
        self._plot()
        self._print_tables()
