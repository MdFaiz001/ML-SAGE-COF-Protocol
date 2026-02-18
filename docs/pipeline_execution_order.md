---

# ML-SAGE-COF Full Pipeline Execution Guide

This document describes the **correct execution order** of all scripts in the repository.

The pipeline proceeds from:

COF generation
→ Structural optimization
→ Pore analysis
→ FLP capacity calculation
→ ML dataset construction

---

# Overview of Pipeline Stages

```
Stage 1  → COF Sampling
Stage 2  → COF Construction (Pormake)
Stage 3  → Structural Filtering
Stage 4  → LAMMPS Interface Preparation
Stage 5  → LAMMPS Geometry Optimization
Stage 6  → Convert Optimized Structures to CIF
Stage 7  → Zeo++ Void Network Analysis
Stage 8  → Extract Global Pore Properties
Stage 9  → Build Model-1 Metadata
Stage 10 → Compute FLP Capacity
Stage 11 → Build Model-2 Metadata
```

---

# Stage 1 — COF Sampling

Script:

```
make_cof_sampling_fixed_final.py
```

Output:

```
COF_generation_plan.csv
```

Purpose:

* Generates combinations of topology, linkers, bases, and functionalization
* Defines the chemical design space

---

# Stage 2 — COF Construction (Pormake)

Script:

```
new_generate_cofs_pormake_timed.py
```

Input:

```
COF_generation_plan.csv
```

Output:

```
generated_cofs_new1/
```

Each folder contains:

```
COF_XXXXXX.cif
```

Also produces:

```
rmsd_log.csv
```

Purpose:

* Builds COF crystal structures
* Evaluates node-topology RMSD feasibility

---

# Stage 3 — Structural Filtering

Script:

```
all_in_one_filter_n1.py
```

Inputs:

* COF_generation_plan.csv
* rmsd_log.csv
* generated_cofs_new1/

Outputs:

* final_cofs_after_rmsd_and_cell_filter.csv
* cof_master_status.csv
* filtered_cofs_ready/

Purpose:

* Removes RMSD-infeasible COFs
* Removes missing CIF builds
* Removes bad unit cells
* Produces clean dataset for optimization

---

# Stage 4 — LAMMPS Interface Preparation

Script:

```
prepare_lammps_inputs_resume_live.py
```

Input:

```
filtered_cofs_ready/
```

Output:

```
cofs_for_optimization/
lammps_interface_success.csv
```

Purpose:

* Converts CIF to LAMMPS data files
* Prepares simulation input files

---

# Stage 5 — LAMMPS Geometry Optimization

Script:

```
launch_lammps_cofs.py
```

Input:

```
cofs_for_optimization/
```

Output:

```
lammps_run_summary.csv
relaxed_COF_XXXXXX.data
```

Purpose:

* Performs geometry optimization
* Generates relaxed atomic coordinates

---

# Stage 6 — Convert Optimized Structures to CIF

Script:

```
data_to_cif.py
```

Input:

```
cofs_for_optimization/
```

Output:

```
COF_CIFs/
conversion_summary.csv
```

Purpose:

* Converts relaxed LAMMPS data files into CIF format

---

# Stage 7 — Zeo++ Void Network Analysis

Script:

```
run_zeo_op_global.py
```

Input:

```
COF_CIFs/
```

Output:

```
COF_XXXXXX.res
ZEO_Task_Report.csv
```

Purpose:

* Generates Voronoi void network (.res files)

---

# Stage 8 — Extract Global Pore Properties

Script:

```
global_prop_ext.py
```

Input:

```
COF_CIFs/
final_cof_ml.csv
```

Output:

```
global_props_opt.csv
```

Purpose:

* Extracts LCD and PLD from Zeo++ results

---

# Stage 9 — Build Model-1 Metadata

Script:

```
build_cof_meta_model1.py
```

Inputs:

* final_cof_ml.csv
* global_props_opt_clean.csv
* descriptor CSV files
* embed/ topology embeddings

Output:

```
cof_meta_model1.csv
```

Purpose:

* Builds dataset for Model-1 (geometric predictor)

---

# Stage 10 — Compute FLP Capacity

Script:

```
flp_batch_all_in_one_n1.py
```

Input:

```
COF_CIFs/
```

Output (per COF):

```
COF_XXXXXX_FLP_large_summary.txt
COF_XXXXXX_FLP_large_sites.csv
COF_XXXXXX_FLP_visual_pockets_large.json
```

Purpose:

* Identifies FLP pockets
* Computes FLP capacity metrics
* Computes simultaneous non-overlapping capacity

---

# Stage 11 — Build Model-2 Metadata

Script:

```
make_model2_meta_flp_large.py
```

Inputs:

* cof_meta_model1_clean.csv
* COF_CIFs/FLP summaries

Output:

```
cof_meta_model2_flp_large.csv
```

Purpose:

* Constructs ML-ready dataset for FLP capacity prediction

---

# Complete Execution Order (Short Version)

Run scripts in this exact order:

```
1. make_cof_sampling_fixed_final.py
2. new_generate_cofs_pormake_timed.py
3. all_in_one_filter_n1.py
4. prepare_lammps_inputs_resume_live.py
5. launch_lammps_cofs.py
6. data_to_cif.py
7. run_zeo_op_global.py
8. combine_master_with_lammps_interface.py
9. cof_ml.py
10. global_prop_ext.py
11. build_cof_meta_model1.py
12. flp_batch_all_in_one_n1.py
13. make_model2_meta_flp_large.py
```

---

# Final Outputs

Model-1 Dataset:

```
cof_meta_model1.csv
```

Model-2 Dataset:

```
cof_meta_model2_flp_large.csv
```

These are the final ML-ready datasets.

---

# Notes

* All steps are deterministic.
* Intermediate outputs are preserved.
* Resume-safe scripts are used for expensive calculations.
* Structural filtering is fully documented.
* FLP capacity is computed using physically motivated void-network criteria.

---

