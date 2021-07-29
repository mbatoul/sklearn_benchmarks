import json
import os

import pandas as pd
from IPython.display import Markdown, display
from sklearn_benchmarks.config import ENV_INFO_PATH, TIME_REPORT_PATH


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
        sklearn_vs_sklearnex="`scikit-learn` vs. `scikit-learn-intelex` (Intel® oneAPI) benchmarks",
        sklearn_vs_onnx="`scikit-learn` vs. `ONNX Runtime` (Microsoft) benchmarks",
        gradient_boosting="Gradient boosting: randomized HPO benchmarks",
    )
    file_extension = "html" if os.environ.get("RESULTS_BASE_URL") else "ipynb"
    display(Markdown("## Notebooks"))
    for file, title in notebook_titles.items():
        display(Markdown(f"### [{title}]({base_url}{file}.{file_extension})"))
