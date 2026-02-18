
# `build_cof_meta_model1.py` — Build `cof_meta_model1.csv` (final metadata table for Model-1 training/inference)

## Purpose

This script creates a single machine-learning ready table:

✅ **`cof_meta_model1.csv`**

Each row corresponds to one COF and includes:

* **Targets:** `LCD`, `PLD`
* **Topology identity:** `topology_name`, `topology_id`, and the **path** to its topology graph embedding file (`.pt`)
* **Building block identity (integer IDs):**

  * node1 linker ID (3C/4C)
  * node2 linker ID (3C/4C)
  * parent (unfunctionalized) 2C linker ID
  * optional functionalized edge linker ID (or `-1` if none)
  * optional base ID (or `-1` if none)
* **Functionalization metadata:** `coverage_pct`, `coverage_fraction`, `edge_linker_type`, `functionalization_type`
* **Other design descriptors:** `bridge_type`, optional `fn_sites`

This file is the **final join-point** that lets you train Model-1 (or run inference) consistently using:

* topology embeddings from `embed/{topology_id}.pt`
* descriptor tables indexed by integer linker IDs
* a unified COF row definition

---

## Inputs required (all must be in the run directory unless paths are edited)

### A) COF list (ML universe)

```text
final_cof_ml.csv
```

This is produced earlier by `cof_ml.py` and must contain **at least** these columns:

`cof_id, topology_name, num_node_types, node1_type, node1_linker, node2_type, node2_linker, parent_2c, edge_fn_name, coverage_pct, bridge_type, base_id`

The script explicitly validates these columns and raises an error if any are missing. 

### B) Targets table (clean Zeo++ properties)

```text
global_props_opt_clean.csv
```

Must contain:

* `cof_id`
* `LCD_A`
* `PLD_A`

These are renamed to `LCD` and `PLD` inside the script, then inner-merged with `final_cof_ml.csv`. COFs missing targets are dropped by the inner merge. 

### C) Topology mapping table

```text
topo_final.csv
```

Must contain:

* `name` (topology string label)
* `topology_id` (integer ID)

This defines the mapping used to locate the graph embedding file. 

### D) Descriptor tables (name → linker_id maps)

These CSVs provide the mapping from a **human-readable building block name** to an **integer ID** that your ML pipeline uses:

* `2_con_fn_linker_feat_ms.csv`   (functionalized 2C)
* `2_con_unfn_linker_feat_ms.csv` (parent/unfunctionalized 2C)
* `3_con_linker_feat_ms.csv`      (3C node linkers)
* `4_con_linker_feat_ms.csv`      (4C node linkers)
* `base_feat_ms.csv`              (base molecules)

Each must contain:

* `name_x` (string name key)
* `linker_id` (int ID)

If required columns are missing, the script raises a clear KeyError. 

### E) Topology embedding directory

```text
embed/
  ├─ 0.pt
  ├─ 1.pt
  ├─ ...
```

This folder must contain `{topology_id}.pt` files.
If a topology `.pt` file is missing for a COF, that COF is dropped (`missing_pt`). 

---

## Output produced

### `cof_meta_model1.csv`

One row per COF that successfully passes all mapping/availability checks.

It prints:

* ✅ number of COFs written
* a “Dropped counts” dictionary summarizing why rows were excluded
* output column list

---

## Core helpers (important for robustness)

### `norm_key(x)`

Normalizes string keys for consistent mapping across CSVs:

* converts NaN / None / empty strings → None
* strips whitespace
* converts `"4.0"` → `"4"` (to protect against float-cast IDs in CSVs)

This avoids common pipeline breakage when IDs are accidentally stored as float strings.

### `parse_connectivity(node_type)`

Converts:

* `"3C"` → `3`
* `"4C"` → `4`
  Also allows `"3"` or `"4"`.

If a weird value is encountered (e.g., `"TETRA"`), it raises a ValueError.

### `build_map(feature_csv, name_col="name_x", id_col="linker_id")`

Loads one descriptor CSV and builds a dictionary:

```text
{name_x → linker_id}
```

Used for:

* node linkers
* parent 2C
* functionalized 2C
* base

---

## Step-by-step logic (exact)

### 1) Load core tables

* Reads `final_cof_ml.csv` into `cof`
* Reads `global_props_opt_clean.csv` into `props`
* Reads `topo_final.csv` into `topo`
* Builds `topo_map: topology_name → topology_id`

### 2) Build all ID maps

Creates dictionaries:

* `fn2c_map`   : functionalized 2C name → linker_id
* `unfn2c_map` : unfunctionalized 2C name → linker_id
* `l3c_map`    : 3C linker name → linker_id
* `l4c_map`    : 4C linker name → linker_id
* `base_map`   : base name → linker_id

### 3) Merge targets (LCD/PLD)

Keeps only the columns:

* `cof_id`, `LCD_A`, `PLD_A`
  Renames:
* `LCD_A → LCD`
* `PLD_A → PLD`

Then merges:

```python
cof = cof.merge(props, on="cof_id", how="inner")
```

So: if a COF has no Zeo++ targets → it is excluded.

### 4) Build meta rows (main loop)

For each COF row:

#### a) Resolve `topology_id`

* normalize topology name
* lookup in `topo_map`
* if missing → drop (`missing_topo_id`)

#### b) Check topology embedding exists

* expects: `embed/{topology_id}.pt`
* if missing → drop (`missing_pt`)

#### c) Resolve node1 ID

* parse `node1_type` → connectivity 3 or 4
* choose mapping:

  * 3 → `l3c_map`
  * 4 → `l4c_map`
* lookup `node1_linker`
* if missing → drop (`missing_node1`)

#### d) Resolve node2 ID

If `num_node_types == 2`:

* parse `node2_type`, map by connectivity
* lookup `node2_linker`
  Else (`num_node_types == 1`):
* node2 is set equal to node1 (mirroring)

If node2 missing → drop (`missing_node2`)

#### e) Resolve parent 2C ID (mandatory)

* `parent_2c` is looked up in `unfn2c_map`
* if missing → drop (`missing_parent_2c`)

#### f) Resolve functionalized edge 2C ID (optional)

* reads `coverage_pct` as float
* reads `edge_fn_name`

Rules:

* if `coverage_pct == 0` OR `edge_fn_name` is empty:

  * `edge_fn_id = -1`
  * `edge_linker_type = "unfn"`
* else:

  * lookup `edge_fn_name` in `fn2c_map`
  * if missing → drop (`missing_edge_fn`)
  * else:

    * `edge_linker_type = "fn"`

#### g) Resolve base ID (optional)

* looks up `base_id` in `base_map`
* if missing: sets `base_id = -1`
  (Notice: it does **not** drop COFs missing base.)

#### h) Infer `functionalization_type`

* if no functionalization → `"none"`
* else:

  * if base exists → `"base"`
  * else → `"functionalized"`
    (You can refine this later if you encode acid/base categories separately.)

#### i) Append final row

It stores:

* identifiers & paths
* integer IDs
* coverage as pct and fraction
* targets LCD/PLD

### 5) Remove duplicate `cof_id` and write CSV

```python
meta = pd.DataFrame(rows).drop_duplicates(subset=["cof_id"])
meta.to_csv("cof_meta_model1.csv", index=False)
```

### 6) Print summary

The script prints:

* number of COFs written
* dropped counts dictionary
* final column list

The `dropped` dict is extremely useful for debugging dataset attrition due to missing descriptor rows or missing embedding files.

---

## Output columns (what appears in `cof_meta_model1.csv`)

Key columns include:

* `cof_id`
* `topology_name`, `topology_id`, `topology_pt_path`
* `node1_connectivity`, `node1_linker_id`
* `node2_connectivity`, `node2_linker_id`
* `parent_2c_id`
* `edge_fn_id` (`-1` if none)
* `edge_linker_type` (`fn` or `unfn`)
* `coverage_pct`, `coverage_fraction`
* `bridge_type`
* `base_id` (`-1` if none)
* `fn_sites` (optional column; may be blank if not present)
* `functionalization_type`
* `LCD`, `PLD`  (targets)

---

## How to run

From the directory containing all required CSVs and `embed/` folder:

```bash
python build_cof_meta_model1.py
```

---

## Notes / limitations 

1. **Inner merge on targets**

   * COFs with missing LCD/PLD are removed automatically.

2. **Strict requirement for descriptor availability**

   * Missing linker name→ID mapping drops a COF (node1/node2/parent_2c/edge_fn).

3. **Embedding availability is mandatory**

   * If `embed/{topology_id}.pt` is missing, that COF is dropped.

4. **Functionalized edge is optional only when coverage = 0**

   * If coverage > 0 but no functionalized ID found → COF dropped.

---

