import glob
import importlib
import os
import re

import numpy as np
from joblib import Memory
from sklearn_benchmarks.config import RESULTS_PATH

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
