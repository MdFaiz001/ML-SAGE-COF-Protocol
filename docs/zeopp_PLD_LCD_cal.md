# `run_zeo_op_global.py` — Parallel Zeo++ “network -res” runner for all optimized COF CIFs + timing report

## Purpose

This script performs the Zeo++ **geometry-only network analysis** for every optimized COF CIF inside `COF_CIFs/` by running:

```bash
network -res <output.res> <input.cif>
```

For each COF, Zeo++ produces a `.res` file which contains the **Voronoi network representation**. From this `.res`, downstream steps can compute pore descriptors such as **PLD/LCD** (and other network-based metrics).

The script is designed to:

* Automatically build a job queue for every `COF_*` folder
* Run Zeo++ jobs in parallel using `multiprocessing.Pool`
* Suppress Zeo++ stdout/stderr (clean console)
* Save a timing report `ZEO_Task_Report.csv`

---

## Inputs required

### 1) Optimized CIF root directory

Configured as:

```python
CIF_ROOT = "./COF_CIFs"
```

Expected structure:

```text
COF_CIFs/
└─ COF_007370/
   └─ COF_007370.cif
```

This directory is typically produced by your previous converter script (`data_to_cif.py`).

### 2) Zeo++ executable on PATH

This script calls Zeo++ via the command:

```bash
network ...
```

So **`network` must be available in your PATH** (e.g., Zeo++ compiled and added to environment).

---

## Outputs produced

### 1) One `.res` file per COF

For each folder `COF_XXXXXX`, the script writes:

```text
COF_CIFs/COF_XXXXXX/COF_XXXXXX.res
```

Because it executes:

```bash
network -res <path>/<folder>.res <cif>
```

Example:

```text
COF_CIFs/COF_007370/COF_007370.res
```

### 2) Global timing report

Writes:

```text
ZEO_Task_Report.csv
```

Columns:

* `cof` → folder name (e.g., `COF_007370`)
* `time_sec` → elapsed wall time for that Zeo++ run (seconds)

---

## Exact processing logic (step-by-step)

### Step 1 — Build global task queue (RES-only)

The script loops through:

```python
for folder in sorted(os.listdir(CIF_ROOT)):
```

It only processes folders that start with:

```python
if not folder.startswith("COF_"):
    continue
```

Inside each `COF_*` folder:

* It searches for `.cif` files:

  ```python
  cif_files = [x for x in os.listdir(path) if x.endswith(".cif")]
  ```
* If none found, skips that folder.

It selects the **first CIF file** in that folder:

```python
cif = os.path.join(path, cif_files[0])
```

Then it creates one task tuple:

```python
(folder, ["network", "-res", f"{path}/{folder}.res", cif])
```

So tasks list contains:

* COF folder name
* The Zeo++ command list to run

At the end it prints:

* total number of tasks generated
* number of workers used (`TOTAL_CORES`)

---

## Parallel execution model

Configured:

```python
TOTAL_CORES = 40
```

It starts:

```python
with mp.Pool(TOTAL_CORES) as pool:
    for cof, dt in tqdm(pool.imap_unordered(worker, tasks), ...):
```

### Important details:

* `imap_unordered` means results come back **as soon as they finish**, not in original order.
* This maximizes throughput because fast COFs don’t wait for slow ones.

---

## Worker function behavior

Each worker does:

1. start timer
2. run Zeo++ command via `subprocess.run(...)`
3. suppress stdout/stderr
4. returns `(cof, elapsed_time_seconds)`

### Critical note about error handling

The helper `run(cmd)` is:

```python
try:
    subprocess.run(..., check=True)
except:
    pass
```

So:

* If Zeo++ fails, it is silently ignored.
* The worker still returns a time.
* There is **no explicit success/failure column** in the output report.

**Implication:** The only way to verify success is to check whether:

* `COF_CIFs/COF_XXXXXX/COF_XXXXXX.res` exists and is non-empty.

(If you want, we can upgrade this script to log `status=ok/error` and stderr text—very useful for publication-grade robustness.)

---

## Output writing

After pool completes:

It writes `ZEO_Task_Report.csv`:

```python
w.writerow(["cof","time_sec"])
for cof, dt in results:
    w.writerow([cof, dt])
```

Then prints:

```
========= RUN COMPLETE =========
```

---

## Expected folder structure

Before:

```text
COF_CIFs/
└─ COF_007370/
   └─ COF_007370.cif
```

After:

```text
COF_CIFs/
└─ COF_007370/
   ├─ COF_007370.cif
   └─ COF_007370.res

ZEO_Task_Report.csv
```

---

## How to run

From project root (where `COF_CIFs/` exists):

```bash
python run_zeo_op_global.py
```

---



* “Zeo++ network analysis (`network -res`) was executed in parallel across optimized COF CIFs to generate the Voronoi network representation (`.res`) required for subsequent pore descriptors such as PLD and LCD. Wall-clock time per structure was recorded in `ZEO_Task_Report.csv`.” 

---

