

# `all_in_one_filter_n1.py` — Full COF filtering pipeline + master status table (single script)

## Purpose

This script takes the **full COF design set** (from `COF_generation_plan.csv`) and the **generated CIF outputs** (from `generated_cofs_new1/`) and performs a complete, automated filtering pipeline:

1. **RMSD feasibility filter** (from `rmsd_log.csv`)
2. **CIF existence check** (missing CIF interpreted as Pormake timeout / build failure)
3. **Extract unit cell parameters + atom count directly from CIF**
4. **Cell geometry quality filter** (length, anisotropy, angle range, and allowed angle patterns)
5. **Sort final COFs by atom count** (smallest first)
6. **Copy passing COF directories to `filtered_cofs_ready/`** (LAMMPS-ready staging)
7. **Create a master status CSV** for *all planned COFs* describing why each passed/failed

It outputs:

* `final_cofs_after_rmsd_and_cell_filter.csv`  (only the pass-all COFs)
* `cof_master_status.csv`                      (every planned COF with statuses + failure reasons)
* `filtered_cofs_ready/`                       (folders ready for LAMMPS interface)

---

## Inputs required

### Required files (must exist in run directory)

* `COF_generation_plan.csv`  *(the plan with all COF IDs and output_dir)*
* `rmsd_log.csv`             *(from the PORMAKE generation stage; node RMSD + feasible flags)*
* `generated_cofs_new1/`     *(the directory containing COF folders and CIFs)*

### Required directory conventions

Each COF is expected at:

```
generated_cofs_new1/<output_dir>/<cof_id>.cif
```

Where:

* `output_dir` is a folder name like `dia__COF_000123`
* `cof_id` is like `COF_000123`

---

## User settings (top of script)

You can adjust:

* `OUTPUT_ROOT = "./generated_cofs_new1"`
* `PLAN_CSV    = "COF_generation_plan.csv"`
* `RMSD_CSV    = "rmsd_log.csv"`

Outputs:

* `FINAL_OUT   = "final_cofs_after_rmsd_and_cell_filter.csv"`
* `MASTER_OUT  = "cof_master_status.csv"`
* `COPY_ROOT   = "./filtered_cofs_ready"`

Angle tolerances:

* `TOL90  = 5.0` degrees
* `TOL120 = 5.0` degrees

---

# Step-by-step pipeline logic

## 1) CIF parameter extraction: `extract_cell_parameters(cif_path)`

For every CIF it reads:

### Extracts:

* `_cell_length_a` → `a`
* `_cell_length_b` → `b`
* `_cell_length_c` → `c`
* `_cell_angle_alpha` → `alpha`
* `_cell_angle_beta` → `beta`
* `_cell_angle_gamma` → `gamma`

### Atom counting logic (important)

The script counts atoms by scanning the CIF atom loop:

* It sets `in_atom_block = True` when it sees the line containing:

  ```
  _atom_type_partial_charge
  ```
* Then it counts all subsequent non-empty lines as atoms, until it sees a line starting with:

  ```
  loop_
  ```
* Each non-empty line with at least two tokens increments `n_atoms`.

So `n_atoms` reflects the number of atom records in the CIF “atom section”.

**Output of this function:**
`(a, b, c, alpha, beta, gamma, n_atoms)`

---

## 2) RMSD feasibility filter: `compute_rmsd_status()`

### Input: `rmsd_log.csv`

The script:

* Reads `rmsd_log.csv`
* Normalizes feasibility values so that strings like `"TRUE"` become boolean `True`:

```python
df["feasible"] = df["feasible"].astype(str).str.upper().eq("TRUE")
```

### Feasibility rule (1-node vs 2-node)

For each `cof_id` group:

* If the group has only one `node_type_idx`:

  * **1-node topology**
  * feasible if `feasible` for that node is True

* Else (more than one `node_type_idx`):

  * **>1-node topology**
  * feasible only if **ALL** node types are feasible (logical AND)

This produces two sets:

* `feasible_ids` = COFs that pass RMSD feasibility
* `rmsd_ids_all` = all COFs that appear in `rmsd_log.csv`

These sets are later used for master status.

---

## 3) CIF existence check: `existing_cif_ids(df_plan)`

For every COF in the plan:

It checks:

```
generated_cofs_new1/<output_dir>/<cof_id>.cif
```

If the CIF exists:

* add cof_id to `existing_ids`
* store path in `cif_paths[cof_id]`

**Interpretation used later:**
If CIF is missing → treated as “timeout / not generated”.

---

## 4) Initial filter intersection

Only COFs satisfying BOTH are considered for cell analysis:
initial_ids=feasible_ids∩existing_ids

So:

* If a COF fails RMSD → not cell-checked
* If CIF missing → not cell-checked

---

## 5) Cell geometry filter

### Helper functions

* `near_90(x)` returns True if `|x - 90| <= 5°`
* `near_120(x)` returns True if `|x - 120| <= 5°`

### Allowed angle patterns: `cell_angle_pattern(alpha,beta,gamma)`

A COF passes if it matches one of the following:

#### Pattern A: Orthorhombic-like

* all angles near 90:

  * α≈90, β≈90, γ≈90

#### Pattern B: Monoclinic-like

* exactly two angles near 90
* the remaining non-90 angle must be:

  * 60° ≤ angle ≤ 130°

#### Pattern C: Hexagonal-like

* α≈90 and β≈90 and γ≈120

### Cell filter core rules: `cell_filter(...)`

A COF passes if:

1. **Minimum cell length**

* `min(a,b,c) >= 10 Å`

2. **Anisotropy constraint**

* `max(a,b,c)/min(a,b,c) <= 10`

3. **Angle sanity**

* each angle must satisfy:

  * 60° ≤ angle ≤ 140°

4. **Angle pattern match**

* must satisfy one of the three patterns above

If any cell parameter is missing (`None`), cell fails with reason:

* `missing_cell_params`

If it fails the filter rules, cell fails with reason:

* `bad_cell`

---

## 6) Final output table and sorting

All COFs passing RMSD + CIF existence + cell filter are collected into `final_rows`.

Then:

* converted to DataFrame
* sorted by `n_atoms` ascending (smallest first)
* written to:

### ✅ `final_cofs_after_rmsd_and_cell_filter.csv`

This CSV contains:

* all original plan columns (copied from plan)
* plus:

  * `a b c alpha beta gamma n_atoms cif_path`

---

## 7) Copy passing COF directories

For each passing COF:

Copy entire directory tree:

From:

```
generated_cofs_new1/<output_dir>
```

To:

```
filtered_cofs_ready/<output_dir>
```

Behavior:

* If destination exists → it is deleted (`shutil.rmtree`) and copied fresh (`shutil.copytree`)

This ensures the staging folder always contains only the most up-to-date filtered COFs.

---

# 8) Master status CSV (most important for reporting)

This script creates a full per-COF status table for **every row in `COF_generation_plan.csv`**, even if never generated.

### Output:

✅ `cof_master_status.csv`

For each COF, it computes:

## A) `rmsd_status`

* `"ok"` if cof_id in `feasible_ids`
* `"fail"` if cof_id in `rmsd_ids_all` but not feasible
* `"not_found"` if cof_id has no entry in `rmsd_log.csv`

## B) `cif_status`

* `"ok"` if CIF exists
* `"missing"` if CIF missing
  (explicitly interpreted as timeout / never built)

## C) `cell_status`

* `"ok"` if cell passed
* `"fail"` if cell failed
* `"not_applicable"` if cell was never checked (because RMSD failed or CIF missing)

## D) `final_status`

* `"ok"` only if:

  * rmsd_status = ok
  * cif_status  = ok
  * cell_status = ok
* otherwise `"fail"`

## E) `failure_reason` (priority logic)

If final_status is fail, reason is assigned by priority:

1. If `rmsd_status == "fail"` → `rmsd_fail`
2. Else if (`rmsd_status == "not_found"` and `cif_status == "missing"`) → `timeout_missing_cif`
3. Else if `rmsd_status == "not_found"` → `no_rmsd_entry`
4. Else if `cif_status == "missing"` → `timeout_missing_cif`
5. Else if `cell_status == "fail"` → `cell_reason` (either `missing_cell_params` or `bad_cell`)
6. Else → `unknown`

It also stores:

* a,b,c,alpha,beta,gamma,n_atoms,cif_path (or NaN/blank if not applicable)

This master file is what you should use to report:

* how many failed at each stage
* why they failed
* and the distribution of cell parameters for passed structures

---

## Outputs summary

After running, you will have:

1. `final_cofs_after_rmsd_and_cell_filter.csv`
2. `cof_master_status.csv`
3. `filtered_cofs_ready/` containing only fully filtered COF folders

---

## How to run

From the directory containing:

* `COF_generation_plan.csv`
* `rmsd_log.csv`
* `generated_cofs_new1/`

Run:

```bash
python all_in_one_filter_n1.py
```

---

## Notes / limitations 

1. **Missing CIF is treated as timeout**

   * This includes Pormake timeouts or any failure to write CIF.

2. **Atom count extraction assumes `_atom_type_partial_charge` marker exists**

   * If CIF format differs (no such column), atom counting may be incorrect.

3. **Cell pattern filter is intentionally restrictive**

   * It accepts only orthorhombic-like / monoclinic-like / hexagonal-like patterns with tolerances.

---
