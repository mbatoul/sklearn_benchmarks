from pathlib import Path

if __name__ == "__main__":
    threshold_in_bytes = 1e8  # 100mb
    profiling_results_path = (
        Path(__file__).resolve().parent.parent / "results" / "profiling"
    )
    files = list(profiling_results_path.glob("*.html"))
    for file in files:
        if file.stat().st_size >= threshold_in_bytes:
            print(f"removing {file} (size {file.stat().st_size} bytes)...")
            file.unlink()
