
## 1) What this GA repository does (big picture)

This project implements a **base-fixed genetic algorithm (GA)** to explore the design space of **base-grafted 3D COFs** (topology + node linkers + edge-bundle functionalization + coverage), and ranks candidates using a **multi-model ML surrogate stack**:

* **Model-1** predicts **LCD, PLD**

* **Model-2 v2** predicts **C_FLP_sim** (your FLP capacity surrogate)

In the GA itself, **Model-1 and Model-1b act as hard feasibility filters**, while **Model-2 drives ranking/selection among feasible genomes**.

There are two operational modes in the code you shared:

1. **GA evolution loop** (pure GA + ML predictions each generation)
   → `run_ga_evolution_base_fixed_fixed.py` 

2. **End-to-end “pipeline runner”** that takes a GA population, generates CIFs with PORMAKE, filters structures, runs LAMMPS-interface optimization, and then predicts all ML targets
   → `ga_streamlined_runner_resume.py` 

The predictor used in both contexts is:
→ `predict_all_models_ga_population.py` 

---

Structure

  ga_streamlined_runner_resume.py
  run_ga_evolution_base_fixed_fixed.py
  predict_all_models_ga_population.py

  # required at runtime (copied/symlinked into generation folders)
  new_generate_cofs_pormake_timed.py
  all_in_one_filter_n1.py
  prepare_lammps_inputs_resume_live.py

  # libraries / metadata
  topo_final.csv
  topo_sorted.csv
  2c_linkers.csv
  3c_linkers.csv
  4c_linkers.csv
  functionalized_2c_linkers.csv
  edge_bundle_library.csv

  # feature tables used by datasets
  features/
    2_con_fn_linker_feat_ms.csv
    2_con_unfn_linker_feat_ms.csv
    3_con_linker_feat_ms.csv
    4_con_linker_feat_ms.csv
    base_feat_ms.csv

  # xyz libraries used by PORMAKE
  2c_xyz/
  3c_xyz/
  4c_xyz/
  fn2c_xyz/

  # topology embeddings (PyTorch Geometric graphs)
  embed/
    <topology_id>.pt

  # trained checkpoints
  model1_final.pt

  model2_v2_flp_large_final.pt

  # model/dataset definitions required by predictor imports
  dataset.py
  model.py

  dataset_model2_v2.py
  model_model2_v2.py

  # LAMMPS template
  in.COF
```

That layout matches what the runner **copies/symlinks** into each generation directory.

---

 ## 3) Genome definition (what a “candidate COF” is)

Each GA individual (genome) is represented as a row in a CSV / dict with the following core genes:

### A) Topology + node linkers

* `topology_id`
* `node1_connectivity`, `node2_connectivity` (derived from topology, not “free genes”)
* `node1_linker_id`, `node2_linker_id` (chosen from pools consistent with CN)

Topology metadata is loaded from `topo_final.csv` and the script **parses connectivities** from the `"Node info"` string (expects substrings like `CN=3`, `CN=4`). 

### B) Edge functionalization bundle (base-fixed)

These genes are treated as a **bundle** that must remain consistent:

* `parent_2c_id`
* `edge_fn_id`
* `bridge_type`
* `base_id` (fixed for the run)

Valid bundles come from `edge_bundle_library.csv`, filtered to `base_id == args.base_id`. Children are repaired to ensure their chosen `(edge_fn_id, parent_2c_id, bridge_type)` exists for that base. 

### C) Coverage

* `coverage_fraction` ∈ {0.25, 0.50, 0.75, 1.0}

**Coverage=0 is intentionally disallowed** for this base-fixed GA. 

### Derived / repaired fields (not free genes)

During “repair”, the script enforces:

* `topology_pt_path = embed/<topology_id>.pt`
* `edge_slots`, `node_slots`, `total_slots` from topology specs
* expected edge counts:

  * `n_fn_edges_expected`, `n_unfn_edges_expected` from `edge_slots` × `coverage_fraction`

This ensures the genome is always **structurally consistent** with topology constraints. 

---

## 4) Fitness function (how the GA ranks candidates)

This is the most important part and your code is very explicit:

### Hard feasibility constraints (filters)

A candidate is “valid” only if:

* `PLD_pred ≥ pld_min`


(`pld_min` default 12.0, ) 

### Optimization objective among valid candidates

For valid genomes:

* **raw fitness** = `C_FLP_ranking` (Model-2 v2 prediction)

Then fitness is **scaled to [0,1] within each generation** among valid genomes:

* If scores differ: `(raw - min) / (max - min)`
* If all equal: all valid get 1.0
* Invalid genomes always get fitness = 0.0 

This design is intentional: it keeps selection stable and avoids giant negative penalties.

---

## 5) Selection, elitism, crossover, mutation (how evolution happens)

### Selection

* `selection_fraction` default 0.25
* elites = top `ceil(selection_fraction * N)` by fitness (after ranking)

### Elitism

* keep `max(2, 0.10*N)` of best elites unchanged into next generation 

### Crossover & mutation (child creation)

Child is built from three gene-groups:

#### Group A: Topology + node linker IDs

* topology chosen from parent1 or parent2 (50/50)
* node linker IDs inherited from one parent (50/50)
* **validated against topology-required CN**:

  * CN=3 → node must be from `3_con_linker_feat_ms.csv` pool
  * CN=4 → node must be from `4_con_linker_feat_ms.csv` pool
* if invalid → resampled CN-safe from pool
* if topology has 1 node type → node2 mirrored to node1
* mutation: with probability `mutation_rate` (default 0.05), resample node IDs CN-safely 

#### Group B: Edge bundle genes (base-fixed)

* bundle inherited from one parent (50/50)
* repaired if bundle tuple is not present in base-specific bundle library
* mutation: with probability `mutation_rate`, replace with a random valid bundle from library 

#### Group C: Coverage

* inherited 50/50
* snapped to nearest allowed value if not allowed
* mutation: with probability `mutation_rate`, choose random from allowed coverage set 

Finally, **repair_genome()** is applied to enforce topology-derived fields, pt path, slots, and expected edge counts.

---

## 6) What each script does

### A) `run_ga_evolution_base_fixed_fixed.py` — the GA engine

**Purpose:** run base-fixed GA for `generations` and save per-generation populations + predictions + ranked tables. 

**Inputs (CLI):**

* `--base_id` (required)
* `--gen0_csv` initial population (required)
* `--topo` path to `topo_final.csv` (required)
* `--edge_bundle` path to `edge_bundle_library.csv` (required)
* `--feat_dir` path containing feature CSVs (required)
* `--embed_dir` topology .pt directory (required)
* `--predictor_py` predictor script path (required)
* `--out_dir` output folder for generations (required)

**Key knobs:**

* `--generations` (default 150)
* `--pld_min` (default 12.0)
* `--av_min` (default 0.30)
* `--selection_fraction` (default 0.25)
* `--mutation_rate` (default 0.05)
* `--seed` (default 42)
* `--device`, `--batch_size` passed to predictor (though predictor itself defines embed usage)

**Outputs (per generation `gen_XXX/`):**

* `population.csv` (genomes before prediction)
* `population_with_preds.csv` (after predictor)
* `population_ranked.csv` (fitness, ranks, validity flags)
* plus it also writes next generation’s `population.csv` early (crash-friendly behavior) 

**Important behavior:**

* This GA script itself is not “marker-based resumable”, but it writes generation outputs incrementally, and pre-writes next gen’s `population.csv`, which makes recovery easier.

---

### B) `predict_all_models_ga_population.py` — multi-model predictor

**Purpose:** given a population CSV, run all three models and append predictions. 

**Inputs (CLI):**

* `--pop_csv` population CSV
* `--embed_dir` directory containing topology `.pt` graphs
* `--out_csv` output file
* `--device` cpu/cuda
* `--batch_size`

**What it reads (assumed in current working directory):**

* feature CSVs at fixed paths:

  * `features/2_con_fn_linker_feat_ms.csv`
  * `features/2_con_unfn_linker_feat_ms.csv`
  * `features/3_con_linker_feat_ms.csv`
  * `features/4_con_linker_feat_ms.csv`
  * `features/base_feat_ms.csv`
* model checkpoints in CWD:

  * `model1_final.pt`

  * `model2_v2_flp_large_final.pt`
* dataset/model python modules in CWD (imports):

  * `dataset.py`, `model.py`, etc. 

**Output columns added:**

* `LCD_pred`, `PLD_pred`

* `C_FLP_ranking`

**Note on target transforms:**
Model-2 v2 supports `target_mode` in checkpoint; if `log1p`, predictions are inverse-transformed with `expm1`. 

---

### C) `ga_streamlined_runner_resume.py` — resumable pipeline runner

**Purpose:** run the *full structural pipeline* for a population in a generation directory, with stage markers for safe resume. 

It performs these stages:

1. **PLAN**
   Builds `COF_generation_plan.csv` from `population_csv` by joining:

   * `topo_final.csv`
   * linker libraries (2c/3c/4c)
   * `edge_bundle_library.csv`

   It maps node linkers to “name” fields from linker CSVs, and writes `output_dir = "<Topology>__<cof_id>"`. 

2. **PORMAKE build**
   Runs `new_generate_cofs_pormake_timed.py` (external) and expects outputs under:

   * `generated_cofs_new1/`
   * `generated_cofs_new1/rmsd_log.csv` (required)
   * optionally `generated_cofs_new1/coverage_log.csv`

   Copies `rmsd_log.csv` up to generation root as `gen_dir/rmsd_log.csv`. 

3. **FILTER**
   Runs `all_in_one_filter_n1.py` (external) and expects:

   * `final_cofs_after_rmsd_and_cell_filter.csv`
   * optionally `cof_master_status.csv` 

4. **LAMMPS interface**
   Runs `prepare_lammps_inputs_resume_live.py` (external) and expects:

   * `lammps_interface_success.csv`
   * `lammps_interface_errors.csv` (optional)
     Requires `in.COF` in the gen directory. 

5. **PREDICT**
   Filters population to only those with successful LAMMPS-interface status (`cof_id` in success CSV), writes:

   * `population_for_ml.csv`

   Then runs:

   * `predict_all_models_ga_population.py`
     producing:
   * `population_with_preds.csv`

**Resumability mechanism:**

* Each stage writes a marker file in `gen_dir`:

  * `.done_plan`, `.done_pormake`, `.done_filter`, `.done_lammps`, `.done_predict`
* On rerun, if marker exists (and key outputs exist), stage is skipped.
* You can override with `--force_stage plan|pormake|filter|lammps|predict|all`. 

**Logging:**

* all subprocess output is tee’d into `ga_pipeline.log` in the generation folder. 

---

## 7) How to run (copy-paste examples)

### Run GA evolution (Model-only GA loop)

```bash
python run_ga_evolution_base_fixed_fixed.py \
  --base_id 7 \
  --gen0_csv populations/base7_gen0.csv \
  --topo topo_final.csv \
  --edge_bundle edge_bundle_library.csv \
  --feat_dir features \
  --embed_dir embed \
  --predictor_py predict_all_models_ga_population.py \
  --out_dir GA_runs/base7 \
  --generations 150 \
  --pld_min 12.0 \
  --av_min 0.30 \
  --selection_fraction 0.25 \
  --mutation_rate 0.05 \
  --device cpu \
  --batch_size 256 \
  --seed 42
```

(Exact args reflect your parser.)

### Run full pipeline for a given generation population (resumable)

```bash
python ga_streamlined_runner_resume.py \
  --gen_dir GA_runs/base7/gen_012 \
  --population_csv GA_runs/base7/gen_012/population.csv \
  --scripts_dir . \
  --embed_dir embed \
  --device cpu \
  --batch_size 256
```

Rerun the same command safely; it resumes using `.done_*` markers. 

---

## 8) Practical considerations 

### Why “repair” is critical

Your GA allows topology to change during crossover. Without repair, topology-dependent genes would become inconsistent (wrong CN, wrong node ID pool, wrong slots, wrong pt path). Your `repair_genome()` enforces feasibility **before every evaluation and after breeding**, which is the correct strategy for constrained materials design search. 

### Why edge “bundle-safe” evolution matters

Edge functionalization is chemically coupled: `parent_2c_id`, `edge_fn_id`, `bridge_type` must match a valid entry in `edge_bundle_library.csv`. Your child creation explicitly repairs invalid tuples by resampling a valid bundle (base-specific), preventing unbuildable candidates early. 

### Why fitness is rank-scaled

Using min–max scaling among valid candidates prevents large magnitude drift and keeps selection pressure consistent across generations, even if Model-2 output distribution shifts. Invalids are cleanly separated by a feasibility flag rather than harsh penalties. 

### Crash safety

* GA engine writes each generation outputs and even pre-writes next gen population file.
* Full pipeline runner uses explicit stage markers and log tee’ing for deterministic resume/debug.

---

