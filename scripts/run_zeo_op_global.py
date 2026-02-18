#!/usr/bin/env python3
import os, csv, time, subprocess, multiprocessing as mp
from tqdm import tqdm

# =====================================================
CIF_ROOT   = "./COF_CIFs"   # test first
RESULT_CSV = "ZEO_Task_Report.csv"

TOTAL_CORES = 40     # total machine threads to use
# =====================================================

def run(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        pass


def worker(task):
    """Executes one Zeo++ job from queue"""
    cof, cmd = task
    start = time.time()
    run(cmd)
    return cof, round(time.time() - start, 2)


# =====================================================
# 🔥 Build global task queue (ONLY .res tasks)
# =====================================================

tasks = []

for folder in sorted(os.listdir(CIF_ROOT)):
    if not folder.startswith("COF_"):
        continue

    path = os.path.join(CIF_ROOT, folder)
    cif_files = [x for x in os.listdir(path) if x.endswith(".cif")]
    if not cif_files:
        continue

    cif = os.path.join(path, cif_files[0])
    base = os.path.splitext(cif_files[0])[0]

    # geometry-only (PLD/LCD come from .res)
    tasks.append((folder, ["network", "-res", f"{path}/{folder}.res", cif]))

print(f"\n🔥 Tasks generated: {len(tasks)} (RES only)")
print(f"Using {TOTAL_CORES} workers\n")

# =====================================================
# 🚀 Parallel Execution (No nesting, No daemon issues)
# =====================================================

results = []
with mp.Pool(TOTAL_CORES) as pool:
    for cof, dt in tqdm(pool.imap_unordered(worker, tasks),
                       total=len(tasks), dynamic_ncols=True):
        results.append((cof, dt))

# =====================================================
# Write summary CSV
# =====================================================
with open(RESULT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cof", "time_sec"])
    for cof, dt in results:
        w.writerow([cof, dt])

print("\n========= RUN COMPLETE =========")
