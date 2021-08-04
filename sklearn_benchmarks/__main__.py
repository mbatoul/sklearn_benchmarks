"""
The main entry point. Invoke as `sklearn_benchmarks' or `python -m sklearn_benchmarks'.
"""
import json
import time
from importlib.metadata import version

import click
import joblib
import pandas as pd
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info
from threadpoolctl import threadpool_info

from sklearn_benchmarks.benchmarking import Benchmark
from sklearn_benchmarks.config import (
    BENCH_LIBS,
    DEFAULT_CONFIG,
    ENV_INFO_PATH,
    TIME_REPORT_PATH,
    VERSIONS_PATH,
    get_full_config,
    parse_parameters,
)
from sklearn_benchmarks.utils.misc import clean_results, convert


@click.command()
@click.option(
    "--append",
    "--a",
    is_flag=True,
    required=False,
    default=False,
    help="Append benchmark results to existing ones.",
)
@click.option(
    "--config",
    "--c",
    type=str,
    default=DEFAULT_CONFIG,
    help="Path to config file.",
)
@click.option(
    "--profiling",
    "--p",
    type=click.Choice(["html", "json.gz"], case_sensitive=True),
    default=["html", "json.gz"],
    multiple=True,
    help="Profiling output formats.",
)
@click.option(
    "--estimator",
    "--e",
    type=str,
    multiple=True,
    help="Estimator to benchmark.",
)
def main(append, config, profiling, estimator):
    if not append:
        clean_results()
    config = get_full_config(config)
    benchmarking_config = config["benchmarking"]
    if not "estimators" in benchmarking_config:
        return

    all_estimators = benchmarking_config["estimators"]
    selected_estimators = all_estimators
    if estimator:
        selected_estimators = {k: all_estimators[k] for k in estimator}

    time_report = pd.DataFrame(
        columns=["estimator", "benchmarking_method", "hour", "min", "sec"]
    )
    t0 = time.perf_counter()

    for name, params in selected_estimators.items():
        # When inherit param is set, we fetch params from parent estimator
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = all_estimators[params["inherit"]]
            # ONNX predictions are run on scikit-learn's estimators only
            if "predict_with_onnx" in params:
                params.pop("predict_with_onnx")
            params["estimator"] = curr_estimator

        for i in range(len(params["datasets"])):
            common_dataset_name = params["datasets"][i].get("name", None)
            if common_dataset_name is not None:
                params["datasets"][i] = benchmarking_config["common_datasets"][
                    common_dataset_name
                ]

        params = parse_parameters(params)

        params["random_seed"] = benchmarking_config.get("random_seed", None)
        params["profiling_output_extensions"] = profiling

        benchmark = Benchmark(**params)
        start_benchmark = time.perf_counter()
        benchmark.run()
        end_benchmark = time.perf_counter()

        time_report.loc[len(time_report)] = [
            name,
            params["benchmarking_method"],
            *convert(end_benchmark - start_benchmark),
        ]

    # Store bench time report
    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", None, *convert(t1 - t0)]
    time_report.to_csv(
        str(TIME_REPORT_PATH),
        mode="w+",
        index=False,
    )

    # Store bench libs versions
    versions = {}
    for lib in BENCH_LIBS:
        versions[lib] = version(lib)
    with open(VERSIONS_PATH, "w") as outfile:
        json.dump(versions, outfile)

    # Store bench environment information
    env_info = {}
    env_info["system_info"] = _get_sys_info()
    env_info["dependencies_info"] = _get_deps_info()
    env_info["threadpool_info"] = threadpool_info()
    env_info["cpu_count"] = joblib.cpu_count(only_physical_cores=True)
    with open(ENV_INFO_PATH, "w") as outfile:
        json.dump(env_info, outfile)


if __name__ == "__main__":
    main()
