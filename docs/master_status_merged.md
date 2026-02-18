# `combine_master_with_lammps_interface.py` — Merge structural QC master table with LAMMPS-interface success to produce final unified status

## Purpose

This script combines two outputs from previous pipeline stages:

1. **`cof_master_status.csv`**
   (produced by `all_in_one_filter_n1.py`: includes RMSD feasibility, CIF existence, and unit-cell quality results for *all planned COFs*)

2. **`lammps_interface_success.csv`**
   (produced by `prepare_lammps_inputs_resume_live.py`: contains only COFs that successfully ran `lammps-interface` and produced valid `data.COF_XXXXXX` and `in.COF_XXXXXX` outputs)

It produces:

✅ **`cof_master_status_final.csv`**
A single **unified pipeline status table** where each COF has:

* `rmsd_status`
* `cif_status`
* `cell_status`
* **`lammps_status`**
* `final_status`
* `failure_reason`

This file is what you should use in the thesis/article to report:

* total attrition at each stage
* final number of fully-prepared COFs that are ready for LAMMPS optimization and later Zeo++/ML/GA steps
* why structures are eliminated (RMSD mismatch vs missing CIF vs bad cell vs lammps-interface failure)

---

## Inputs required

### 1) Structural QC master table

```text
cof_master_status.csv
```

Must contain at least these columns:

* `cof_id`
* `rmsd_status` (expected values: `ok`, `fail`, `not_found`)
* `cif_status`  (expected: `ok`, `missing`)
* `cell_status` (expected: `ok`, `fail`, `not_applicable`)

### 2) LAMMPS interface success log

```text
lammps_interface_success.csv
```

Must contain at least:

* `cof_id`

If this file is missing, the script assumes **no COFs succeeded** at lammps-interface stage and sets all `lammps_status = fail`.

---

## Output produced

### Unified final master table

```text
cof_master_status_final.csv
```

It contains **all original columns** from `cof_master_status.csv` plus:

* `lammps_status`
* `final_status`
* `failure_reason` *(recomputed)*

---

## Exact logic implemented

### Step 1 — Load master table

```python
df = pd.read_csv(MASTER_IN)
df["cof_id"] = df["cof_id"].astype(str)
```

Ensures `cof_id` is treated as a string (prevents `COF_000001` becoming numeric).

---

### Step 2 — Load LAMMPS success list

```python
df_succ = pd.read_csv(LAMMPS_SUCC)
succ_ids = set(df_succ["cof_id"].tolist())
```

If file missing:

* prints warning:

  ```
  [WARN] lammps_interface_success.csv not found. Assuming none successful.
  ```
* sets `succ_ids = empty set`

---

### Step 3 — Assign `lammps_status`

For every `cof_id` in the master table:

* If `cof_id ∈ succ_ids` → `lammps_status = "ok"`
* Else → `lammps_status = "fail"`

This is a **binary status**: the script does not import detailed failure modes (like NO_CIF, EXCEPTION) from `lammps_interface_log.csv`. It intentionally treats everything not in success list as a fail.

---

### Step 4 — Compute `final_status` and `failure_reason` (priority logic)

A COF gets `final_status = "ok"` only if **all four** stage statuses are ok:

* `rmsd_status == ok`
* `cif_status  == ok`
* `cell_status == ok`
* `lammps_status == ok`

Otherwise `final_status = "fail"` and failure reason is assigned by priority:

1. if `rmsd_status != ok` → `failure_reason = "rmsd_fail"`
2. else if `cif_status != ok` → `failure_reason = "missing_cif"`
3. else if `cell_status != ok` → `failure_reason = "bad_cell"`
4. else if `lammps_status != ok` → `failure_reason = "lammps_fail"`
5. else → `failure_reason = "unknown"`

This means each COF receives exactly **one** dominant failure reason.

---

## Why this file matters (for thesis / journal)

This script gives you a **single auditable data table** that can be used to generate:

* Sankey/flowchart of attrition
* bar plots: number of COFs failing at each stage
* final “ready for optimization” dataset size
* fairness and robustness claims (because selection criteria are explicit and reproducible)

It is the “pipeline bookkeeping” script.

---

## How to run

Run from the directory containing the two input CSV files:

```bash
python combine_master_with_lammps_interface.py
```

Expected terminal output:

* If success file exists:

  ```
  [INFO] Unified master file written to: cof_master_status_final.csv
  [INFO] DONE.
  ```
* If success file missing:

  ```
  [WARN] lammps_interface_success.csv not found. Assuming none successful.
  [INFO] Unified master file written to: cof_master_status_final.csv
  [INFO] DONE.
  ```

---

## Folder structure (typical)

```text
project_root/
├─ cof_master_status.csv
├─ lammps_interface_success.csv
├─ combine_master_with_lammps_interface.py
└─ cof_master_status_final.csv     (generated)
```

---


