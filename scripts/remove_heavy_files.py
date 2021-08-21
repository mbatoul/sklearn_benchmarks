import os

threshold_in_bytes = 1e8  # 100mb

files = os.listdir("./results/profiling")
for file in files:
    if os.path.getsize(file) >= threshold_in_bytes:
        print(f"removing {file}...")
        os.remove(file)
