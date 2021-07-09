import os
from pathlib import Path

import yaml

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"
PROFILING_RESULTS_PATH = RESULTS_PATH / "profiling"
BENCHMARKING_RESULTS_PATH = RESULTS_PATH / "benchmarking"
TIME_REPORT_PATH = RESULTS_PATH / "time_report.csv"
ENV_INFO_PATH = RESULTS_PATH / "env_info.txt"
VERSIONS_PATH = RESULTS_PATH / "versions.txt"
DEFAULT_CONFIG = "config.yml"
BASE_LIB = "sklearn"
FUNC_TIME_BUDGET = 30
BENCHMARK_MAX_ITER = 10
SPEEDUP_COL = "mean"
STDEV_SPEEDUP_COL = "stdev"
PLOT_HEIGHT_IN_PX = 350
REPORTING_FONT_SIZE = 12
DEFAULT_COMPARE_COLS = [SPEEDUP_COL, STDEV_SPEEDUP_COL]
BENCHMARK_TIME_BUDGET = 300
BENCH_LIBS = ["scikit-learn", "scikit-learn-intelex", "xgboost", "lightgbm", "catboost"]
HPO_CURVES_COLORS = ["blue", "red", "green", "purple", "orange"]


def get_full_config(config=None):
    if config is None:
        config = os.environ.get("DEFAULT_CONFIG")
    with open(config, "r") as config_file:
        config = yaml.full_load(config_file)
    return config


def prepare_params(params):
    from sklearn_benchmarks.utils.misc import is_scientific_notation

    init_params = params.get("hyperparameters", {}).get("init", {})
    for key, value in init_params.items():
        if not isinstance(value, list):
            continue
        for i, el in enumerate(value):
            if is_scientific_notation(el):
                if "-" in el:
                    init_params[key][i] = float(el)
                else:
                    init_params[key][i] = int(float(el))

    datasets = params.get("datasets", [])
    for dataset in datasets:
        dataset["n_features"] = int(float(dataset["n_features"]))
        for i, ns_train in enumerate(dataset["n_samples_train"]):
            dataset["n_samples_train"][i] = int(float(ns_train))
        for i, ns_test in enumerate(dataset["n_samples_test"]):
            dataset["n_samples_test"][i] = int(float(ns_test))

    return params
