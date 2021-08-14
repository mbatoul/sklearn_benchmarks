import glob
import importlib
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from joblib import Memory

from sklearn_benchmarks.config import (
    ENV_INFO_PATH,
    RESULTS_PATH,
    TIME_REPORT_PATH,
    COMPARABLE_COLS,
)


def print_time_report():
    df = pd.read_csv(str(TIME_REPORT_PATH), index_col="estimator")
    df = df.sort_values(by=["hour", "min", "sec"])

    df["sec"] = df["sec"].round(0)
    df[["hour", "min", "sec"]] = df[["hour", "min", "sec"]].astype(int)

    row_total, df = df.iloc[-1], df.iloc[:-1]
    total_hour, total_min, total_sec = row_total[["hour", "min", "sec"]].values
    subtitle = f"Total time elapsed: {total_hour}h {total_min}m {total_sec}s"

    display(Markdown("## Time report"))
    display(Markdown(f"{subtitle}"))
    for benchmarking_method, df in df.groupby(["benchmarking_method"]):
        benchmarking_method = benchmarking_method.replace("_", " ")
        benchmarking_method = benchmarking_method.capitalize()
        display(Markdown(f"### {benchmarking_method}"))

        df = df.drop(columns=["benchmarking_method"])
        display(df)


def print_env_info():
    with open(ENV_INFO_PATH) as json_file:
        data = json.load(json_file)
    display(Markdown("## Benchmark environment"))
    print(json.dumps(data, indent=2))


def display_links_to_notebooks():
    if os.environ.get("RESULTS_BASE_URL") is not None:
        base_url = os.environ.get("RESULTS_BASE_URL")
    else:
        base_url = "http://localhost:8888/notebooks/"
    notebook_titles = dict(
        scikit_learn_intelex_vs_scikit_learn="scikit-learn-intelex (Intel® oneAPI) vs. scikit-learn benchmarks",
        onnx_vs_scikit_learn="ONNX Runtime (Microsoft) vs. scikit-learn benchmarks",
        gradient_boosting="Gradient boosting: randomized HPO benchmarks",
    )
    file_extension = "html" if os.environ.get("RESULTS_BASE_URL") else "ipynb"
    display(Markdown("## Notebooks"))
    for file, title in notebook_titles.items():
        display(Markdown(f"### [{title}]({base_url}{file}.{file_extension})"))


def gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def make_hover_template(df):
    template = ""
    for index, name in enumerate(df.columns):
        template += "<b>%s</b>: %%{customdata[%i]}<br>" % (name, index)
    template += "<extra></extra>"
    return template


def identify_pareto(data):
    n = data.shape[0]
    all_indices = np.arange(n)
    front_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if data.iloc[i][0] > data.iloc[j][0] and data.iloc[i][1] < data.iloc[j][1]:
                front_pareto[i] = 0
                break
    return all_indices[front_pareto]


def select_front_pareto(point, data_pareto):
    pareto_indices = identify_pareto(data_pareto)
    front_pareto = data_pareto.iloc[pareto_indices].to_numpy()
    point_in_front_pareto = np.any(np.all(point == front_pareto, axis=1))
    return point_in_front_pareto


_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


@memory.cache
def gen_data(
    generator_path, n_samples=1000, n_features=10, random_state=None, **kwargs
):
    """Returns a tuple of data from the specified generator."""
    splitted_path = generator_path.split(".")
    module, func = ".".join(splitted_path[:-1]), splitted_path[-1]
    generator_func = getattr(importlib.import_module(module), func)
    data = generator_func(
        n_samples=n_samples, n_features=n_features, random_state=random_state, **kwargs
    )
    return data


def is_scientific_notation(string):
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def clean_results():
    extensions = [".csv", ".html", ".json.gz", ".txt"]
    files = []
    for extension in extensions:
        files_path = str(RESULTS_PATH / "**/*") + extension
        files += glob.glob(str(files_path), recursive=True)

    for file in files:
        if file.split("/")[-1] == "index.html":
            continue
        try:
            os.remove(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return hour, min, sec


def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_lib_alias(lib):
    aliases = dict(sklearn="scikit-learn", sklearnex="scikit-learn-intelex")
    return aliases.get(lib, lib)


def string_matches_substrings(string, substrings):
    return any(map(string.__contains__, substrings))


def diff_between_lists(l1, l2):
    return list(set(l1) - set(l2)) + list(set(l2) - set(l1))


def get_position(string):
    top_columns = [
        "estimator",
        "n_samples_train",
        "n_samples",
        "n_features",
    ]
    if string in top_columns:
        return 4
    elif "mean_duration" in string:
        return 3
    elif "std_duration" in string:
        return 2
    elif "score" in string:
        return 1
    elif "speedup" in string:
        return 0
    else:
        return -1


class HoverTemplateMaker:
    def __init__(self, df):
        self.df = df
        self.dimensions = ["n_samples_train", "n_samples", "n_features"]

    def __len__(self):
        return len(self.dimensions) + len(self.measurements) + len(self.parameters)

    def __iter__(self):
        return iter([*self.dimensions, *self.measurements, *self.parameters])

    def at(self, i):
        return [*self.dimensions, *self.measurements, *self.parameters][i]

    def split_columns_in_groups(self):
        measurements = filter(
            lambda col: string_matches_substrings(
                col,
                [
                    *COMPARABLE_COLS,
                    "speedup",
                    "std_speedup",
                    "iteration_throughput",
                    "latency",
                ],
            ),
            self.df.columns,
        )
        measurements = sorted(measurements, key=get_position, reverse=True)
        self.measurements = list(measurements)

        parameters = diff_between_lists(
            self.df.columns, [*self.dimensions, *self.measurements]
        )
        parameters = sorted(parameters)
        self.parameters = list(parameters)

    def make_template(self):
        self.split_columns_in_groups()

        titles_indices = {0: "Dimensions"}
        titles_indices[len(self.dimensions)] = "Benchmark measurements"
        titles_indices[len(self.dimensions) + len(self.measurements)] = "Parameters"

        template = ""
        for i in range(len(self)):
            if i in titles_indices:
                template += "<br>"
                template += f"<b>{titles_indices[i]}</b><br>"
            template += "%s: %%{customdata[%i]}<br>" % (self.at(i), i)
        template += "<extra></extra>"

        return template

    def make_data(self):
        return self.df[[*self.dimensions, *self.measurements, *self.parameters]].values
