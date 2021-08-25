from pathlib import Path
import os

if __name__ == "__main__":
    threshold_in_bytes = 1e8  # 100mb
    profiling_results_path = (
        Path(__file__).resolve().parent.parent / "results" / "profiling"
    )
    files = []
    for extension in ["html", "json.gz"]:
        files += list(profiling_results_path.glob(f"*.{extension}"))
    for file in files:
        if file.stat().st_size >= threshold_in_bytes:
            print(f"removing {file} (size {file.stat().st_size} bytes)...")
            os.remove(file)
