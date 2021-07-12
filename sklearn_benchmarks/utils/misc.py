import glob
import importlib
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
    STATIC_FILE_EXTENSIONS,
    REPORTING_NOTEBOOKS_TITLES,
)

_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


@memory.cache
def gen_data(
    generator_path, n_samples=1000, n_features=10, random_state=None, **kwargs
):
    """
    Returns a tuple of data from the specified generator.
    """
    splitted_path = generator_path.split(".")
    module, func = ".".join(splitted_path[:-1]), splitted_path[-1]
    generator_func = getattr(importlib.import_module(module), func)
    data = generator_func(
        n_samples=n_samples, n_features=n_features, random_state=random_state, **kwargs
    )
    return data


def is_scientific_notation(string):
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def predict_or_transform(estimator):
    if hasattr(estimator, "predict"):
        bench_func = estimator.predict
    else:
        bench_func = estimator.transform
    return bench_func


def delete_static_files():
    files = []
    for extension in STATIC_FILE_EXTENSIONS:
        files_path = str(RESULTS_PATH / "**/*") + "." + extension
        files += glob.glob(str(files_path), recursive=True)

    for file in files:
        try:
            os.remove(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return hour, min, sec


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def print_time_report():
    df = pd.read_csv(str(TIME_REPORT_PATH), index_col="algo")
    df = df.sort_values(by=["hour", "min", "sec"])

    display(Markdown("## Time report"))
    for index, row in df.iterrows():
        display(Markdown("**%s**: %ih %im %is" % (index, *row.values)))


def print_env_info():
    with open(ENV_INFO_PATH) as json_file:
        data = json.load(json_file)
    display(Markdown("## Benchmark environment"))
    print(json.dumps(data, indent=2))


def display_links_to_notebooks():
    if os.environ.get("RESULTS_BASE_URL") is not None:
        base_url = os.environ.get("RESULTS_BASE_URL")
        extension = "html"
    else:
        base_url = "http://localhost:8888/notebooks/"
        extension = "ipynb"

    display(Markdown("## Notebooks"))

    for file, title in REPORTING_NOTEBOOKS_TITLES.items():
        display(Markdown(f"[{title}]({base_url}{file}.{extension})"))


def load_dynamically(path):
    splitted_path = path.split(".")
    module, attr = ".".join(splitted_path[:-1]), splitted_path[-1]
    return getattr(importlib.import_module(module), attr)
