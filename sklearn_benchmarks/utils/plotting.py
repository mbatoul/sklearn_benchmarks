import itertools

import numpy as np
import pandas as pd


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


def _compute_cumulated(fit_times, scores):
    cumulated_fit_times = fit_times.cumsum()
    best_val_score_so_far = pd.Series(scores).cummax()
    return cumulated_fit_times, best_val_score_so_far


def boostrap_fit_times(
    fit_times,
    scores,
    n_bootstraps=10_000,
    baseline_score=0.7,
):
    grid_scores = np.linspace(
        baseline_score, scores.max(), 1000
    ) # take max of max and share grid_scores
    all_fit_times = []
    rng = np.random.RandomState(0)
    n_samples = fit_times.shape[0]
    for _ in range(n_bootstraps):
        indices = rng.randint(n_samples, size=n_samples)
        cum_fit_times_p, cum_scores_p = _compute_cumulated(
            fit_times.iloc[indices], scores.iloc[indices]
        )
        grid_fit_times = np.interp(
            grid_scores,
            cum_scores_p,
            cum_fit_times_p,
            right=cum_fit_times_p.max(),
        )
        all_fit_times.append(grid_fit_times)

    return all_fit_times, grid_scores


def percentile_bootstrapped_curve(
    fit_times,
    scores,
    q,
    n_bootstraps=10_000,
    baseline_score=0.7,
):
    fit_times, grid_scores = boostrap_fit_times(
        fit_times,
        scores,
        n_bootstraps=n_bootstraps,
        baseline_score=baseline_score,
    )

    return np.percentile(fit_times, q, axis=0), grid_scores


def mean_bootstrapped_curve(
    fit_times,
    scores,
    n_bootstraps=10_000,
    baseline_score=0.7,
):
    fit_times, grid_scores = boostrap_fit_times(
        fit_times,
        scores,
        n_bootstraps=n_bootstraps,
        baseline_score=baseline_score,
    )

    return np.mean(fit_times, axis=0), grid_scores


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
