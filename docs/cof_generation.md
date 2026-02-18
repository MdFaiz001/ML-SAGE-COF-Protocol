Below is a **GitHub-ready, nothing-missed** documentation for **Code 2**: `new_generate_cofs_pormake_timed.py`. It includes **every input, output, folder, parameter, logic branch, logging file, resume behavior, timeout behavior, functionalization placement algorithm, and failure behavior**. 

---

# `new_generate_cofs_pormake_timed.py` — COF builder (PORMAKE) with timed multiprocessing + functionalization placement + logs

## Purpose

This script **constructs COF structures (CIF files)** from the sampling plan (`COF_generation_plan.csv`) using **PORMAKE** building blocks and topologies. It supports:

* **Parallel generation** (multiprocessing Pool)
* **Per-COF timeout** to prevent hanging builds
* **RMSD feasibility evaluation** for node building blocks (via PORMAKE Locator)
* **Partial edge functionalization** according to `coverage_pct` using a deterministic slot-selection algorithm
* **Resuming** from a given `cof_id`
* Writes **CIF outputs** + **two global log files**:

  * `generated_cofs_new1/rmsd_log.csv`
  * `generated_cofs_new1/coverage_log.csv`

---

## What you must have before running

### 1) Input CSV files (in the working directory)

These files must exist where you run the script (same folder unless you edit paths):

* `COF_generation_plan.csv`  *(design plan from Code 1)*
* `topo_sorted.csv`          *(topology definitions + node/edge info)*
* `2c_linkers.csv`
* `3c_linkers.csv`
* `4c_linkers.csv`
* `functionalized_2c_linkers.csv`

Important: this script assumes specific columns exist in these CSVs:

#### `COF_generation_plan.csv` must contain at least:

* `cof_id` (e.g., `COF_000001`)
* `topology_name`
* `output_dir`
* `num_node_types`
* `node1_type`, `node1_linker`
* `node2_type`, `node2_linker` (for 2-node topologies)
* `parent_2c`
* `edge_fn_name` (may be blank)
* `coverage_pct`

#### Linker CSVs:

* `2c_linkers.csv`, `3c_linkers.csv`, `4c_linkers.csv`, `functionalized_2c_linkers.csv`

  * Must contain columns:

    * `name` (used as index via `.set_index("name")`)
    * `xyz_file` (filename of the building block geometry)

#### `topo_sorted.csv` must contain:

* `Topology`
* `Node info`
* `Edge info`

---

### 2) Input geometry directories (XYZ building blocks)

The script loads building blocks from these directories (relative paths):

* `./2c_xyz/`     → parent 2C edge linkers
* `./3c_xyz/`     → 3C node linkers
* `./4c_xyz/`     → 4C node linkers
* `./fn2c_xyz/`   → functionalized 2C linkers (edge variants)

Each linker’s CSV row points to an `xyz_file`, which must exist inside the corresponding folder.

---

## Outputs produced

### A) Generated COF directories + CIF files

Output root:

* `./generated_cofs_new1/`

For each plan row, the script creates:

* `./generated_cofs_new1/<output_dir>/`

and writes:

* `./generated_cofs_new1/<output_dir>/<cof_id>.cif`

Example:

```text
generated_cofs_new1/
└─ dia__COF_000257/
   └─ COF_000257.cif
```

---

### B) Global log files (written at end)

Two CSV logs are written under `OUTPUT_ROOT`:

1. **`generated_cofs_new1/rmsd_log.csv`**

   * One row **per node type per COF**
   * Columns:

     * `cof_id`
     * `node_type_idx` (0 or 1)
     * `node_CN` (from parsed topology node info)
     * `rmsd` (PORMAKE-calculated RMSD between local structure and building block)
     * `feasible` (True if `rmsd <= RMSD_THRESHOLD`)

2. **`generated_cofs_new1/coverage_log.csv`**

   * One row **per COF**
   * Always includes:

     * `cof_id`
     * `timed_out` (True/False)
   * If `timed_out == False`, also includes:

     * `parent_2c`
     * `edge_fn_name`
     * `coverage_pct_target`
     * `total_edge_slots`
     * `fn_edges_total`
     * `fn_edges_per_type` (JSON string mapping edge-type → count)

> **Important:** the script collects all results in memory and writes logs only at the end. If interrupted mid-run, logs will be lost unless you add incremental logging.

---

## User settings (edit these at top of script)

These parameters control performance and behavior:

* `NUM_PROCESSES = 44`
  Number of parallel worker processes (set based on CPU cores and memory)

* `OUTPUT_ROOT = "./generated_cofs_new1"`
  Main output directory

* `RMSD_THRESHOLD = 0.30`
  Feasibility cutoff for node RMSD evaluation

* `RNG_MASTER_SEED = 2025`
  Master seed for deterministic functionalization placement

* `TIME_LIMIT = 180`
  Seconds allowed per COF build + CIF write **inside each worker**

* `START_FROM_COF_ID = None`
  Resume from a particular COF ID (keeps only remaining rows after that ID)

---

## Full workflow: how each COF is processed

### 1) Main process loads and prepares data

* Reads plan CSV into `plan_rows` list of dicts
* Reads topology and linker tables
* Sets linker tables indexed by `name`

### 2) Resume logic (optional)

If `START_FROM_COF_ID` is set (e.g., `"COF_004501"`):

* Script skips all rows until it encounters that `cof_id`
* Then processes the remainder
* Prints:

  * `[INFO] Resuming from ... remaining: ...`

> Note: this is a **plan-row filter**, not a “skip already-built directories” system.

---

## Worker: `process_one_cof()`

Each worker is fully self-contained and does the following:

### A) Prepare worker-specific PORMAKE objects

Inside worker (important for multiprocessing safety):

* `database = pm.Database()`
* `builder  = pm.Builder()`
* `locator  = pm.Locator()`

### B) Load topology info

* Extract topology node/edge slot information from `topo_sorted.csv`:

  * `parse_node_info()` parses `"type i (CN=..., slots a-b)"` patterns
  * `parse_edge_info()` parses `"(i, j) (slots a-b)"` patterns
* Load actual topology object using PORMAKE:

  * `topo = database.get_topo(topo_name)`

### C) Build node building blocks (from XYZ)

`build_node_bbs_for_row()` constructs node BBs:

* node1 is always assigned to PORMAKE type index `0`
* node2 (if `num_node_types == 2`) is assigned to type index `1`

It converts `"3C"/"4C"` to CN, selects `xyz_file` from:

* `3c_xyz/` if CN=3
* `4c_xyz/` if CN=4

Then creates:

* `pm.BuildingBlock(xyz_path)`

### D) Load edge (2C) building blocks

`get_edge_parent_and_fn_bb()`:

* Always loads parent edge BB from:

  * `2c_xyz/<xyz_file>`
* Loads functionalized edge BB only if:

  * `edge_fn_name` exists and is found in `functionalized_2c_linkers.csv`
  * from `fn2c_xyz/<xyz_file>`

Returns:

* `parent_bb`
* `fn_bb` (or None)

Then creates `edge_bbs_parent = {etype: parent_bb for etype in edge_info}`
i.e., all edge types initially use the unfunctionalized parent.

---

## RMSD evaluation (node feasibility)

Before building the full COF, the script computes RMSD values:

For each node type index (0, and 1 if present):

* `local = topo.unique_local_structures[node_i]`
* `rmsd_val = locator.calculate_rmsd(local, bb)`
* `feasible = rmsd_val <= RMSD_THRESHOLD`

These values are appended into `rmsd_list` and later written into `rmsd_log.csv`.

> The script does **not** stop the build if RMSD is infeasible; it only logs feasibility.

---

## Building the BB slot list

The script converts node BBs + edge BB mapping into a slot-indexed list:

* `bbs = builder.make_bbs_by_type(topo, node_bbs, edge_bbs_parent)`

This returns an array-like `bbs` indexed by slot number.

---

# ✅ Functionalization placement logic (this IS where coverage is implemented)


### Functionalization happens by selecting a subset of edge “slots”

`select_functionalized_slots(edge_types, coverage_pct, rng)`:

1. Collects all slot indices from all edge types
2. Computes target number of functionalized slots:

```text
target = max(1, round(total_slots * coverage_pct / 100))
```

So:

* if coverage > 0, it always selects **at least 1 slot**
* total_slots = number of distinct edge slots across edge types

3. Ensures distribution:

* First pass: chooses **at least 1 random slot per edge type** (if remaining > 0)
* Then fills remaining target slots from the remaining pool randomly

Outputs:

* `fn_slots` (set of slot indices to functionalize)
* `fn_counts` (dict edge-type → number selected)

### Applying functionalized BBs

If `fn_bb` exists:

* for each slot `s in fn_slots`:

  * `bbs[s] = fn_bb.copy()`

So the final COF has a mixture:

* mostly parent 2C linkers
* some functionalized linkers at selected slots

### Deterministic randomness (reproducible)

The RNG seed is tied to each COF:

```text
rng = Random(RNG_MASTER_SEED + hash(cof_id) % 2**31)
```

So for the same cof_id and seed, **the same edge slots will be functionalized**.

---

## Build + CIF write with per-COF timeout

The build + write step is protected by an alarm:

* `signal.alarm(TIME_LIMIT)`
* `MOF = builder.build(topo, bbs)`
* `MOF.write_cif(cif_path)`
* `signal.alarm(0)` (always reset)

If timeout occurs:

* `timed_out = True`
* CIF may not exist

If exception occurs:

* the code sets `timed_out = True` (even if it was not a timeout)

> Note: the script currently treats any exception as “timed_out=True” without recording the exception message in the final coverage log.

---

## Console output during run

The main controller prints:

* `[INFO] Starting multiprocessing with N workers`
* For each completed COF:

  * `[DONE] COF_000123`
* After completion:

  * `[INFO] All COFs processed.`
  * `[INFO] All logs saved.`

---

## Failure behavior (important)

There are 3 failure points:

1. **Node BB loading failure**

   * returns `coverage: {"timed_out": True, "error": str(e)}`
   * However, the final `coverage_log.csv` does NOT record `error` because the writer only keeps `timed_out` when True.

2. **BB list generation failure** (`make_bbs_by_type`)

   * same behavior: timed_out True; error message dropped at final write stage

3. **Build/write failure or timeout**

   * sets `timed_out=True`
   * exception message is not stored

**Implication:** You can identify failures via `timed_out=True`, but you cannot see the reason from the final logs unless you modify script to include error text.

---

## Expected folder structure (recommended)

```text
project_root/
├─ new_generate_cofs_pormake_timed.py
├─ COF_generation_plan.csv
├─ topo_sorted.csv
├─ 2c_linkers.csv
├─ 3c_linkers.csv
├─ 4c_linkers.csv
├─ functionalized_2c_linkers.csv
│
├─ 2c_xyz/
├─ 3c_xyz/
├─ 4c_xyz/
└─ fn2c_xyz/
│
└─ generated_cofs_new1/              (generated)
   ├─ rmsd_log.csv                   (generated at end)
   ├─ coverage_log.csv               (generated at end)
   ├─ <topo>__COF_000001/COF_000001.cif
   ├─ <topo>__COF_000002/COF_000002.cif
   └─ ...
```

---

## How to run

From the directory containing the CSVs and XYZ folders:

```bash
python new_generate_cofs_pormake_timed.py
```

Recommended: run inside a controlled conda environment containing `pormake`, `pandas`.

---


