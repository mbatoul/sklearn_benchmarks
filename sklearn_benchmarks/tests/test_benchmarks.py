from os import listdir

import pandas as pd
from numpy.testing import assert_array_equal
from sklearn_benchmarks.config import BENCHMARKING_RESULTS_PATH


def test_estimators_reach_max_iter():
    """Check that estimators that expose a n_iter_ attribute have reached their max_iter parameter."""

    EXCLUDED_ESTIMATORS = ["LogisticRegression"]

    results_files = [f for f in listdir(BENCHMARKING_RESULTS_PATH) if "csv" in f]

    for result_file in results_files:
        estimator = result_file.split("_")[-1].split(".")[0]

        if estimator in EXCLUDED_ESTIMATORS:
            continue

        df_results = pd.read_csv(f"{BENCHMARKING_RESULTS_PATH}/{result_file}")
        df_results = df_results.dropna(axis=1)

        if not "n_iter" in df_results.columns or not "max_iter" in df_results.columns:
            continue

        n_iters = df_results["n_iter"].values
        max_iters = df_results["max_iter"].values

        assert_array_equal(n_iters, max_iters)
