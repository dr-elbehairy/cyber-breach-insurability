"""Reproduce every result: python run_all.py (about 30-60 minutes; the bootstrap goodness-of-fit tests dominate)."""
import subprocess, sys, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
for s in ["00_download_data.py", "01_data.py", "02_frequency.py", "03_tail.py", "03b_tail_checks.py", "04_accumulation.py", "05_figures.py"]:
    print(">>", s, flush=True); subprocess.run([sys.executable, s], check=True)
for f in os.listdir(HERE):
    if f.startswith("results_") and f.endswith(".json"): shutil.copy(f, os.path.join("results", f))
