import itertools

import numpy as np


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
