from sklearn_benchmarks.config import PROFILING_RESULTS_PATH

threshold_in_bytes = 1e8  # 100mb

files = list(PROFILING_RESULTS_PATH.glob("*.html"))
for file in files:
    if file.stat().st_size >= threshold_in_bytes:
        print(f"removing {file} (size {file.stat().st_size} bytes)...")
        file.unlink()
