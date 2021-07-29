import glob
import importlib
import os
import time
from pprint import pprint

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils._testing import set_random_state
from viztracer import VizTracer

from sklearn_benchmarks.config import (
    BENCHMARKING_RESULTS_PATH,
    FUNC_TIME_BUDGET,
    PROFILING_RESULTS_PATH,
    RESULTS_PATH,
    HPO_TIME_BUDGET,
    HPO_PREDICTIONS_TIME_BUDGET,
    BENCHMARKING_METHODS_N_EXECUTIONS,
)
from sklearn_benchmarks.utils.misc import gen_data, predict_or_transform


class BenchFuncExecutor:
    """
    Executes a benchmark function (fit, predict or transform)
    """

    def run(
        self,
        func,
        estimator,
        profiling_output_path,
        profiling_output_extensions,
        X,
        y=None,
        n_executions=10,
        onnx_model_filepath=None,
        **kwargs,
    ):
        if n_executions > 1:
            # First run with a profiler (not timed)
            with VizTracer(verbose=0) as tracer:
                tracer.start()
                if y is not None:
                    func(X, y, **kwargs)
                else:
                    func(X, **kwargs)
                tracer.stop()
                for extension in profiling_output_extensions:
                    output_file = f"{profiling_output_path}.{extension}"
                    tracer.save(output_file=output_file)

        # Next runs: at most n_executions runs or 30 sec total execution time
        times = []
        start = time.perf_counter()
        for _ in range(n_executions):
            start = time.perf_counter()

            if y is not None:
                self.func_result_ = func(X, y, **kwargs)
            else:
                if onnx_model_filepath is not None:
                    sess = rt.InferenceSession(onnx_model_filepath)
                    input_name = sess.get_inputs()[0].name
                    label_name = sess.get_outputs()[0].name

                    self.func_result_ = sess.run(
                        [label_name], {input_name: X.astype(np.float32)}
                    )[0]
                else:
                    self.func_result_ = func(X, **kwargs)

            end = time.perf_counter()
            times.append(end - start)

            if end - start > FUNC_TIME_BUDGET:
                break

        benchmark_info = {}
        mean = np.mean(times)

        n_iter = None
        if hasattr(estimator, "n_iter_"):
            benchmark_info["n_iter"] = estimator.n_iter_
            n_iter = estimator.n_iter_
        elif hasattr(estimator, "best_iteration_"):
            n_iter = estimator.best_iteration_
        elif hasattr(estimator, "get_best_iteration"):
            n_iter = estimator.get_best_iteration()
        elif hasattr(estimator, "get_booster"):
            n_iter = estimator.get_booster().best_iteration

        benchmark_info["n_iter"] = n_iter
        n_iter = 1 if n_iter is None else n_iter

        benchmark_info["mean_duration"] = mean
        benchmark_info["std_duration"] = np.std(times)
        benchmark_info["throughput"] = X.nbytes * n_iter / mean / 1e9
        benchmark_info["latency"] = mean / X.shape[0]

        return benchmark_info


class Benchmark:
    def __init__(
        self,
        name="",
        estimator="",
        inherit=False,
        metrics=[],
        hyperparameters={},
        datasets=[],
        predict_with_onnx=False,
        onnx_params={},
        random_state=None,
        benchmarking_method="",
        profiling_file_type="",
        profiling_output_extensions=[],
    ):
        self.name = name
        self.estimator = estimator
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets
        self.predict_with_onnx = predict_with_onnx
        self.onnx_params = onnx_params
        self.random_state = random_state
        self.benchmarking_method = benchmarking_method
        self.profiling_file_type = profiling_file_type
        self.profiling_output_extensions = profiling_output_extensions

    def _make_params_grid(self):
        params = self.hyperparameters.get("init", {})
        if not params:
            estimator_class = self._load_estimator_class()
            estimator = estimator_class()
            # Parameters grid should have list values
            params = {k: [v] for k, v in estimator.__dict__.items()}
        grid = list(ParameterGrid(params))
        np.random.shuffle(grid)
        return grid

    def _set_lib(self):
        self.lib_ = self.estimator.split(".")[0]

    def _load_estimator_class(self):
        split_path = self.estimator.split(".")
        mod, class_name = ".".join(split_path[:-1]), split_path[-1]
        return getattr(importlib.import_module(mod), class_name)

    def _load_metrics_funcs(self):
        module = importlib.import_module("sklearn.metrics")
        return [getattr(module, m) for m in self.metrics]

    def run(self):
        self._set_lib()
        estimator_class = self._load_estimator_class()
        metrics_funcs = self._load_metrics_funcs()
        params_grid = self._make_params_grid()
        self.results_ = []
        start = time.perf_counter()
        for dataset in self.datasets:
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = list(reversed(sorted(dataset["n_samples_test"])))
            n_samples_valid = dataset.get("n_samples_valid", None)
            for ns_train in n_samples_train:
                n_samples = ns_train + max(n_samples_test)
                if n_samples_valid is not None:
                    n_samples += n_samples_valid
                X, y = gen_data(
                    dataset["sample_generator"],
                    n_samples=n_samples,
                    n_features=n_features,
                    **dataset["params"],
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=ns_train, random_state=self.random_state
                )
                if n_samples_valid is not None:
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X_train,
                        y_train,
                        test_size=n_samples_valid,
                        random_state=self.random_state,
                    )
                fit_params = {}
                for k, v in self.hyperparameters.get("fit", {}).items():
                    fit_params[k] = eval(str(v))

                for params in params_grid:
                    estimator = estimator_class(**params)
                    set_random_state(estimator, random_state=self.random_state)
                    bench_func = estimator.fit
                    # Use digests to identify results later in reporting
                    hyperparams_digest = joblib.hash(params)
                    dataset_digest = joblib.hash(dataset)
                    profiling_output_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_fit_{hyperparams_digest}_{dataset_digest}"

                    benchmark_info = BenchFuncExecutor().run(
                        bench_func,
                        estimator,
                        profiling_output_path,
                        self.profiling_output_extensions,
                        X_train,
                        y=y_train,
                        n_executions=1,
                        **fit_params,
                    )

                    if self.predict_with_onnx:
                        initial_type = [
                            ("float_input", FloatTensorType([None, X_train.shape[1]]))
                        ]
                        onnx_model_filepath = f"{RESULTS_PATH}/{self.lib_}_{self.name}_{hyperparams_digest}_{dataset_digest}.onnx"

                        onx = convert_sklearn(estimator, initial_types=initial_type)

                        with open(onnx_model_filepath, "wb") as f:
                            f.write(onx.SerializeToString())

                    row = dict(
                        estimator=self.name,
                        function=bench_func.__name__,
                        n_samples_train=ns_train,
                        n_samples=ns_train,
                        n_features=n_features,
                        hyperparams_digest=hyperparams_digest,
                        dataset_digest=dataset_digest,
                        **benchmark_info,
                        **params,
                    )

                    self.results_.append(row)

                    start_predictions = time.perf_counter()
                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = predict_or_transform(estimator)

                        profiling_output_path = f"{PROFILING_RESULTS_PATH}/{self.lib_}_{bench_func.__name__}_{hyperparams_digest}_{dataset_digest}"
                        executor = BenchFuncExecutor()
                        bench_func_params = (
                            self.hyperparameters[bench_func.__name__]
                            if bench_func.__name__ in self.hyperparameters
                            else {}
                        )

                        n_executions = BENCHMARKING_METHODS_N_EXECUTIONS[
                            self.benchmarking_method
                        ]
                        if self.predict_with_onnx:
                            benchmark_info = executor.run(
                                bench_func,
                                estimator,
                                profiling_output_path,
                                self.profiling_output_extensions,
                                X_test_,
                                n_executions=n_executions,
                                onnx_model_filepath=onnx_model_filepath,
                                **bench_func_params,
                            )

                            row = dict(
                                estimator=self.name,
                                is_onnx=True,
                                function=bench_func.__name__,
                                n_samples_train=ns_train,
                                n_samples=ns_test,
                                n_features=n_features,
                                hyperparams_digest=hyperparams_digest,
                                dataset_digest=dataset_digest,
                                **benchmark_info,
                                **params,
                            )

                            for metric_func in metrics_funcs:
                                y_pred = executor.func_result_
                                score = metric_func(y_test_, y_pred)
                                row[metric_func.__name__] = score

                            pprint(row)
                            self.results_.append(row)

                        benchmark_info = executor.run(
                            bench_func,
                            estimator,
                            profiling_output_path,
                            self.profiling_output_extensions,
                            X_test_,
                            n_executions=n_executions,
                            **bench_func_params,
                        )

                        row = dict(
                            estimator=self.name,
                            is_onnx=False,
                            function=bench_func.__name__,
                            n_samples_train=ns_train,
                            n_samples=ns_test,
                            n_features=n_features,
                            hyperparams_digest=hyperparams_digest,
                            dataset_digest=dataset_digest,
                            **benchmark_info,
                            **params,
                        )

                        for metric_func in metrics_funcs:
                            y_pred = executor.func_result_
                            score = metric_func(y_test_, y_pred)
                            row[metric_func.__name__] = score

                        pprint(row)
                        self.results_.append(row)
                        self.to_csv()

                        if self.benchmarking_method == "hpo":
                            now = time.perf_counter()
                            if now - start > HPO_TIME_BUDGET:
                                if self.predict_with_onnx:
                                    os.remove(onnx_model_filepath)
                                return
                            if now - start_predictions > HPO_PREDICTIONS_TIME_BUDGET:
                                break

                    if self.predict_with_onnx:
                        os.remove(onnx_model_filepath)
        return self

    def to_csv(self):
        csv_path = f"{BENCHMARKING_RESULTS_PATH}/{self.lib_}_{self.name}.csv"
        pd.DataFrame(self.results_).to_csv(
            csv_path,
            mode="w+",
            index=False,
        )
