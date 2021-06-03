import itertools
import re
import pandas as pd
import numpy as np

from sklearn_benchmarks.config import DEFAULT_COMPARE_COLS


def gen_coordinates_grid(n_rows, n_cols):
    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))
    return coordinates


def order_columns(columns):
    bottom_cols = DEFAULT_COMPARE_COLS

    def order_func(col):
        for bottom_col in bottom_cols:
            pattern = re.compile(f"^({bottom_col})\_.*")
            if pattern.search(col):
                return 1
        return -1

    return sorted(columns, key=lambda col: order_func(col))


def make_hover_template(df):
    columns = order_columns(df.columns)
    template = ""
    for index, name in enumerate(columns):
        template += "%s: <b>%%{customdata[%i]}</b><br>" % (name, index)
    template += "<extra></extra>"
    return template


def _compute_cumulated(fit_times, scores):
    cumulated_fit_times = fit_times.cumsum()
    best_val_score_so_far = pd.Series(scores).cummax()
    return cumulated_fit_times, best_val_score_so_far


def percentile_permutated_curve(
    fit_times, cum_scores, q, n_permutations=1000, baseline_score=0.7
):
    grid_scores = np.linspace(baseline_score, cum_scores.max(), 1000)
    all_fit_times = []
    rng = np.random.RandomState(0)
    for _ in range(n_permutations):
        indices = rng.permutation(fit_times.shape[0])
        cum_fit_times_p, cum_scores_p = _compute_cumulated(
            fit_times.iloc[indices], cum_scores.iloc[indices]
        )
        grid_fit_times = np.interp(
            grid_scores,
            cum_scores_p,
            cum_fit_times_p,
            right=cum_fit_times_p.max(),
        )
        all_fit_times.append(grid_fit_times)
    return np.percentile(all_fit_times, q, axis=0), grid_scores
