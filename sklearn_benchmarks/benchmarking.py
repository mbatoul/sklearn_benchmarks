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
from sklearn.utils import check_random_state
from sklearn.utils._testing import set_random_state
from viztracer import VizTracer

from sklearn_benchmarks.config import (
    BENCHMARKING_METHODS_N_EXECUTIONS,
    BENCHMARKING_RESULTS_PATH,
    FUNC_TIME_BUDGET,
    HPO_PREDICTIONS_TIME_BUDGET,
    HPO_TIME_BUDGET,
    PROFILING_RESULTS_PATH,
    RESULTS_PATH,
)
from sklearn_benchmarks.utils.misc import gen_data


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
        start_global = time.perf_counter()
        for _ in range(n_executions):
            start_iter = time.perf_counter()

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

            end_iter = time.perf_counter()
            times.append(end_iter - start_iter)

            if end_iter - start_global > FUNC_TIME_BUDGET:
                break

        benchmark_info = {}
        mean = np.mean(times)

        benchmark_info["n_iter"] = (
            estimator.n_iter_ if hasattr(estimator, "n_iter_") else None
        )
        benchmark_info["best_iteration"] = None

        if hasattr(estimator, "best_iteration_"):
            benchmark_info["best_iteration"] = estimator.best_iteration_
        elif hasattr(estimator, "get_best_iteration"):
            benchmark_info["best_iteration"] = estimator.get_best_iteration()
        elif hasattr(estimator, "get_booster"):
            benchmark_info["best_iteration"] = estimator.get_booster().best_iteration

        benchmark_info["mean_duration"] = mean
        benchmark_info["std_duration"] = np.std(times)
        benchmark_info["iteration_throughput"] = X.nbytes / mean / 1e9
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
        random_seed=None,
        benchmarking_method="",
        time_budget=HPO_TIME_BUDGET,
        profiling_file_type="",
        profiling_output_extensions=[],
    ):
        self.name = name
        self.estimator = estimator
        self.inherit = inherit
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.datasets = datasets
        self.random_state = check_random_state(random_seed)
        self.predict_with_onnx = predict_with_onnx
        self.benchmarking_method = benchmarking_method
        self.time_budget = time_budget
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
        self.random_state.shuffle(grid)
        return grid

    def _load_estimator_class(self):
        split_path = self.estimator.split(".")
        mod, class_name = ".".join(split_path[:-1]), split_path[-1]
        return getattr(importlib.import_module(mod), class_name)

    def _load_metrics_funcs(self):
        module = importlib.import_module("sklearn.metrics")
        return [getattr(module, m) for m in self.metrics]

    def run(self):
        library = self.estimator.split(".")[0]
        estimator_class = self._load_estimator_class()
        metrics_funcs = self._load_metrics_funcs()
        params_grid = self._make_params_grid()
        benchmark_results = []
        start = time.perf_counter()
        for dataset in self.datasets:
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = sorted(dataset["n_samples_test"], reverse=True)
            for ns_train in n_samples_train:
                n_samples = ns_train + max(n_samples_test)
                X, y = gen_data(
                    dataset["sample_generator"],
                    n_samples=n_samples,
                    n_features=n_features,
                    random_state=self.random_state,
                    **dataset["params"],
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=ns_train, random_state=self.random_state
                )

                for params in params_grid:
                    estimator = estimator_class(**params)
                    set_random_state(estimator, random_state=self.random_state)
                    bench_func = estimator.fit
                    # Use digests to identify results later in reporting
                    hyperparams_digest = joblib.hash(params)
                    dataset_digest = joblib.hash(dataset)
                    profiling_output_path = f"{PROFILING_RESULTS_PATH}/{library}_fit_{hyperparams_digest}_{dataset_digest}"

                    benchmark_info = BenchFuncExecutor().run(
                        bench_func,
                        estimator,
                        profiling_output_path,
                        self.profiling_output_extensions,
                        X_train,
                        y=y_train,
                        n_executions=1,
                    )

                    if self.predict_with_onnx:
                        initial_type = [
                            ("float_input", FloatTensorType([None, X_train.shape[1]]))
                        ]
                        onnx_model_filepath = f"{RESULTS_PATH}/{library}_{self.name}_{hyperparams_digest}_{dataset_digest}.onnx"

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

                    benchmark_results.append(row)

                    start_predictions = time.perf_counter()
                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]
                        bench_func = estimator.predict

                        profiling_output_path = f"{PROFILING_RESULTS_PATH}/{library}_{bench_func.__name__}_{hyperparams_digest}_{dataset_digest}"
                        executor = BenchFuncExecutor()
                        predict_params = (
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
                                **predict_params,
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
                            benchmark_results.append(row)

                        benchmark_info = executor.run(
                            bench_func,
                            estimator,
                            profiling_output_path,
                            self.profiling_output_extensions,
                            X_test_,
                            n_executions=n_executions,
                            **predict_params,
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
                        benchmark_results.append(row)
                        csv_path = (
                            f"{BENCHMARKING_RESULTS_PATH}/{library}_{self.name}.csv"
                        )
                        pd.DataFrame(benchmark_results).to_csv(
                            csv_path,
                            mode="w+",
                            index=False,
                        )

                        if self.benchmarking_method == "hpo":
                            now = time.perf_counter()
                            if now - start > self.time_budget:
                                if self.predict_with_onnx:
                                    os.remove(onnx_model_filepath)
                                return
                            if now - start_predictions > HPO_PREDICTIONS_TIME_BUDGET:
                                break

                    if self.predict_with_onnx:
                        os.remove(onnx_model_filepath)
        return self
