
# `cof_ml.py` — Build the final “ML universe” list by intersecting pipeline-pass COFs with LAMMPS-run presence

## Purpose

Your pipeline produces multiple “status tables” at different stages. This script creates the final list of COFs that are eligible for the **ML dataset universe** by enforcing two conditions:

1. The COF passed all upstream filters and the LAMMPS-interface step
   → `final_status == "ok"` in `cof_master_status_final.csv`

2. The COF is present in the LAMMPS optimization stage summary
   → `cof_id` exists in `lammps_run_summary.csv`

The output is:

✅ `final_cof_ml.csv`
This becomes the “authoritative list” of COFs for which you will extract final properties (e.g., PLD/LCD from Zeo++ `.res`), and for which ML training/evaluation can be done.

---

## Inputs required

### 1) Unified master status file (pipeline QC + lammps-interface)

```python
cof_master_file = "cof_master_status_final.csv"
```

This file is produced by `combine_master_with_lammps_interface.py` and must contain:

* `cof_id`
* `final_status`

`final_status` is expected to be the string `"ok"` for accepted COFs.

### 2) LAMMPS optimization run summary

```python
lammps_summary_file = "lammps_run_summary.csv"
```

This file is produced by `launch_lammps_cofs.py` and must contain:

* `cof_id` (column name must exactly match `COF_ID_COL`, default `"cof_id"`)

> Important: This script checks only “presence in this summary file”, not whether the run succeeded. If you want “success-only”, see the note in the limitations section below. 

---

## Output produced

### `final_cof_ml.csv`

```python
output_file = "final_cof_ml.csv"
```

This is a filtered subset of the master table. It keeps **all columns** from `cof_master_status_final.csv`, but only for COFs meeting both conditions.

---

## Step-by-step logic (exact)

### Step 1 — Read both CSV files

```python
cof_master_df = pd.read_csv(cof_master_file)
lammps_df = pd.read_csv(lammps_summary_file)
```

### Step 2 — Filter master table to only “OK” COFs

```python
cof_master_ok = cof_master_df[
    cof_master_df["final_status"].astype(str).str.strip() == "ok"
].copy()
```

Console print:

```text
[INFO] COFs with final_status == OK: <N>
```

This ensures only COFs that passed:

* RMSD feasibility
* CIF existence
* cell quality filter
* lammps-interface success
  (whatever your master table encodes in `final_status`).

### Step 3 — Extract COF IDs from LAMMPS summary

```python
lammps_cof_ids = set(lammps_df[COF_ID_COL].astype(str).str.strip())
```

Console print:

```text
[INFO] COFs present in LAMMPS summary: <M>
```

This set contains **every COF that appears in the optimization summary** (success or failure).

### Step 4 — Intersection (the key selection)

```python
final_ml_df = cof_master_ok[
    cof_master_ok[COF_ID_COL].astype(str).str.strip().isin(lammps_cof_ids)
].copy()
```

Console print:

```text
[INFO] Final COFs selected for ML universe: <K>
```

So `final_ml_df` includes only COFs that:

* passed upstream QC (`final_status == ok`)
* and had a LAMMPS optimization entry (present in run summary)

### Step 5 — Write output CSV

```python
final_ml_df.to_csv(output_file, index=False)
```

Console print:

```text
[SUCCESS] ML universe file written → final_cof_ml.csv
```

---

## Interpretation: what this file means

A COF present in `final_cof_ml.csv` means:

* It passed all structural-quality filters
* It successfully produced LAMMPS-interface preparation outputs
* It entered the LAMMPS optimization stage (was launched / recorded)

This is your **ML-ready universe** for final property extraction and model development.

---

## Limitations (important to state in docs/thesis)

1. **LAMMPS success is not enforced**
   The current script checks only if the COF appears in `lammps_run_summary.csv`, not whether:

* `status == "success"`
  or whether `relaxed_COF_XXXXXX.data` exists.

So in rare cases, this could include a COF that was launched but failed.

### Recommended improvement (optional)

If you want to ensure optimization success, change Step 3 to:

* filter `lammps_df` to `status == "success"` before building `lammps_cof_ids`

2. **Column name must match**
   It assumes COF id column is exactly `"cof_id"` in both files.

---

## Expected folder context

```text
project_root/
├─ cof_master_status_final.csv
├─ lammps_run_summary.csv
├─ cof_ml.py
└─ final_cof_ml.csv      (generated)
```

---

## How to run

```bash
python cof_ml.py
```

---

