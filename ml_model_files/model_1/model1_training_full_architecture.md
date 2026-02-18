
---

# Model-1 (COFModel1): Full Architecture + Data Feeding + Training (Detailed)

This document explains **exactly** how Model-1 is constructed and trained using the following files:

* `dataset.py` → builds per-COF tensors (multi-modal inputs + targets)
* `model.py` → defines the neural network architecture
* `train.py` → performs dataset splitting, train-only normalization, training loop, evaluation, and model saving

Model-1 predicts **two continuous COF geometric properties** simultaneously:

* `LCD` = Largest Cavity Diameter (Å)
* `PLD` = Pore Limiting Diameter (Å)

---

## 1) What type of ML model is this?

Model-1 is a **deep learning model** implemented in **PyTorch**.

More precisely, it is a **multi-input neural network** where five different feature streams are encoded separately using small MLP encoders, then fused (late fusion) and passed to a regression head to predict `[LCD, PLD]`.

It is NOT:

* a graph neural network (GNN) trained end-to-end on topology graphs
* a transformer
* a classical ML model (RandomForest / XGBoost / linear regression)

But it DOES use a **topology embedding** that was generated elsewhere (stored as `.pt`). The embedding is treated as a fixed numeric vector and fed into an MLP encoder.

---

## 2) What does one training sample look like?

For each COF, the dataset returns a dictionary of tensors:

* `topo`   : topology embedding vector (no normalization)
* `node`   : averaged node linker descriptor vector (normalized; can be masked by feature family)
* `linker` : concatenated 2C parent + 2C functionalized edge descriptors (normalized; can be masked)
* `base`   : base descriptor vector (normalized; can be masked)
* `misc`   : [coverage_fraction + bridge_type one-hot] (no normalization)
* `y`      : target vector `[LCD, PLD]`

This “multi-branch” structure is essential: each part represents a different chemical/structural component of the COF design.

---

# PART A — dataset.py (How features are constructed)

## A1) Why `add_safe_globals([Data])` exists

At the top of `dataset.py`, the code contains:

* `from torch.serialization import add_safe_globals`
* `from torch_geometric.data import Data`
* `add_safe_globals([Data])`

This is needed because newer PyTorch versions may restrict `torch.load()` for unknown classes during deserialization.
Since your topology embeddings may be saved as PyTorch-Geometric `Data` objects, you explicitly allow safe loading of the `Data` class.

Practical meaning: topology `.pt` files that contain PyG graphs will load correctly.

---

## A2) Dataset initialization: what files are loaded?

`COFModel1Dataset(meta_csv, feature_paths, embed_dir, norm_stats=None, feature_family_mode="all")`

### Inputs:

1. `meta_csv`
   A metadata CSV where each row = one COF. It must include at least:

* `topology_pt_path` (path to `.pt` file)
* `node1_connectivity`, `node2_connectivity` (3 or 4)
* `node1_linker_id`, `node2_linker_id` (integer IDs)
* `parent_2c_id` (integer ID)
* `edge_fn_id` (integer ID; may be `-1`)
* `base_id` (integer ID; may be `-1`)
* `bridge_type` (categorical)
* `coverage_fraction` (float)
* `LCD`, `PLD` (targets)

2. `feature_paths`
   A dictionary of CSV paths for descriptor tables:

* `2c_fn`   → functionalized 2C linker feature table
* `2c_unfn` → unfunctionalized 2C linker feature table (parent)
* `3c`      → 3-connected node linker feature table
* `4c`      → 4-connected node linker feature table
* `base`    → base feature table

Each table is loaded and indexed by `linker_id`.

3. `embed_dir`
   Directory of topology embedding `.pt` files.
   Note: in your implementation, the dataset primarily loads `topology_pt_path` directly from the meta CSV (the `embed_dir` is stored but not used in `__getitem__` for path construction — it’s there for protocol consistency).

4. `norm_stats` (optional)
   Train-only normalization mean/std tensors for `node`, `linker`, and `base`.

5. `feature_family_mode`
   Ablation mode for masking feature subsets inside vectors.

---

## A3) Descriptor column selection (dropping non-numeric labels)

After reading feature tables, the dataset defines column lists like:

* `self.l3c_cols`
* `self.l4c_cols`
* `self.unfn2c_cols`
* `self.fn2c_cols`
* `self.base_cols`

For each table it drops `name_x` if present:

* `columns.drop('name_x', errors='ignore')`

Meaning: model uses **only numeric descriptor columns**.

---

## A4) Misc feature encoding: bridge types (fixed vocabulary)

Bridge types are hard-coded as:

```
['none', 'ch2', 'ph', 'dir']
```

A mapping is created:

* `bridge_map = {bridge_type: index}`

During `__getitem__`, `bridge_type` becomes a one-hot vector of length 4.

If your meta CSV contains bridge types outside this list, it will crash (KeyError).
So ensure your pipeline uses exactly these bridge labels.

---

## A5) Feature-family masking (very important, implemented inside the dataset)

### What is feature-family masking?

Your descriptor CSVs include many features, but they naturally group into families:

* `geom_*`  → geometry-related descriptors (prefix `geom_`)
* `RDF_*`   → RDF shell / distribution descriptors (prefix `RDF_`)
* “Mordred” → everything else numeric that is not `geom_*` and not `RDF_*`

The dataset computes index sets for each family for each table.

### How indices are computed

For a column list `cols`, it finds indices:

* `geom_idx` = indices where column name starts with `"geom_"`
* `rdf_idx`  = indices where column name starts with `"RDF_"`
* `mord_idx` = all remaining indices

This is stored in `_family_idx` for:

* `node3` (3C table)
* `node4` (4C table)
* `parent2c` (unfunctionalized 2C)
* `edge2c` (functionalized 2C)
* `base`

### Special handling: linker combined vector

Your `linker` input is concatenation:

* parent_2c features + edge_fn features

So the dataset builds “combined” family indices by shifting edge indices by `+n_parent`.

This is stored as:

* `_family_idx["linker_combined"]`

### Available modes

* `"all"` (default) → no masking
* `"rdf_only"`      → keep only RDF_*, set everything else to 0
* `"geom_only"`     → keep only geom_*, set everything else to 0
* `"mordred_only"`  → keep only non-geom and non-RDF, set others to 0
* `"no_rdf"`        → zero out RDF features
* `"no_geom"`       → zero out geom features
* `"no_mordred"`    → zero out the remaining features

### Important: how masking is applied

Masking is applied by **zeroing entries** in the vector.

This means:

* vector shape remains unchanged
* the model architecture does not need to change
* you can do ablation studies without rebuilding any model code

---

## A6) Vector lookup method (`_get_vec`)

`_get_vec(df, idx, cols)` does:

* If `idx == -1`: return a zero vector of length `len(cols)`
* Else:

  * selects row `df.loc[idx, cols]`
  * converts entries to numeric using `pd.to_numeric(..., errors='coerce')`
  * replaces NaN with 0.0
  * returns torch.float32 tensor

This behavior is important for:

* missing functionalization (edge_fn_id = -1)
* missing base (base_id = -1)

So the model can handle “no base” and “no functionalization” cases consistently.

---

## A7) Normalization inside dataset (`_norm`)

Normalization is applied only if `norm_stats` is provided:

* `node` normalized with `norm_stats['node']`
* `linker` normalized with `norm_stats['linker']`
* `base` normalized with `norm_stats['base']`

Formula:

* `(x - mean) / std`

Std is clamped in `train.py` to avoid division by 0.

No normalization is applied to:

* `topo` vector
* `misc` vector
* targets `y`

---

## A8) **getitem** full construction logic (exact order)

For each COF row:

### Step 1: topology vector (NO normalization)

* load object from `row['topology_pt_path']` using `torch.load(..., weights_only=False)`
* if object has `.x`: `topo_vec = mean(topo_obj.x over nodes)`
* else: `topo_vec = topo_obj`

This produces a float vector.

### Step 2: node vectors (3C vs 4C decision)

If `node1_connectivity == 3`:

* use 3C feature table
  Else:
* use 4C feature table

Same logic for node2.

Then compute:

* `node_vec = 0.5 * (n1 + n2)`
* normalize using `norm_stats['node']` (if provided)
* apply feature-family masking using family indices (mask uses connectivity family indices)

### Step 3: linker vector (parent + edge)

* `parent_2c = _get_vec(unfn2c, parent_2c_id)`
* `edge_fn   = _get_vec(fn2c, edge_fn_id)`  (zero if -1)
* `linker_vec = concat([parent_2c, edge_fn])`
* normalize using `norm_stats['linker']`
* apply masking using `linker_combined` family indices

### Step 4: base vector

* `base_vec = _get_vec(base, base_id)` (zero if -1)
* normalize using `norm_stats['base']`
* apply masking using base family indices

### Step 5: misc vector (NO normalization)

Build:

* coverage tensor: `[coverage_fraction]`
* bridge one-hot: length 4

Concatenate:

* `misc_vec = [coverage_fraction] + one_hot(bridge_type)`

### Step 6: target y

* `y = [LCD, PLD]` float32 tensor

### Return dict

Keys returned exactly:

* `'topo','node','linker','base','misc','y'`

---

# PART B — model.py (Neural network architecture)

## B1) MLP block definition

A reusable MLP block:

* Linear(in_dim -> out_dim)
* ReLU
* Dropout(drop)  where default `drop=0.2`

This is a shallow MLP (one linear layer), but it is used multiple times and the final regressor adds more layers.

---

## B2) COFModel1 architecture (multi-branch encoders + late fusion)

The constructor receives:

* `dims` = dictionary with input dimensions for each stream:

  * dims['topo']
  * dims['node']
  * dims['linker']
  * dims['base']
  * dims['misc']

Default hidden size:

* `hidden = 128`

### Encoders

* topo_enc   : dims['topo']   -> 128
* node_enc   : dims['node']   -> 128
* linker_enc : dims['linker'] -> 128
* base_enc   : dims['base']   -> 128
* misc_enc   : dims['misc']   -> 64  (hidden // 2)

### Fusion (concatenation)

Outputs of all encoders are concatenated along feature dimension:

* fused_dim = 128 + 128 + 128 + 128 + 64 = 576

### Regressor head

A small head maps fused representation to 2 outputs:

* Linear(576 -> 128)
* ReLU
* Dropout(0.2)
* Linear(128 -> 2)

Final output:

* `[pred_LCD, pred_PLD]`

---

## B3) Forward pass expectation

The model forward expects a batch dict containing keys:

* `'topo','node','linker','base','misc'`

Each value must be a 2D tensor:

* shape = (batch_size, feature_dim)

The output shape:

* (batch_size, 2)

---

# PART C — train.py (Splitting, normalization, training, evaluation, saving)

## C1) Reproducibility controls

`seed_everything(42)` sets:

* PYTHONHASHSEED
* Python random seed
* NumPy seed
* torch.manual_seed
* torch.cuda.manual_seed_all
* cudnn deterministic = True
* cudnn benchmark = False

Meaning: runs are reproducible (within typical GPU determinism limits).

---

## C2) Paths (fixed to match your workflow)

Hard-coded protocol paths:

* META  = "cof_meta_model1_clean.csv"
* EMBED = "embed"

Feature descriptor CSVs:

* "features/2_con_fn_linker_feat_ms.csv"
* "features/2_con_unfn_linker_feat_ms.csv"
* "features/3_con_linker_feat_ms.csv"
* "features/4_con_linker_feat_ms.csv"
* "features/base_feat_ms.csv"

This file layout must exist for training to run.

---

## C3) Training hyperparameters

Defined constants:

* BATCH_SIZE = 16
* LR         = 1e-3
* WEIGHT_DECAY = 0.0
* MAX_EPOCHS = 2000
* PATIENCE   = 25

Loss function:

* torch.nn.MSELoss() over two outputs

Optimizer:

* Adam(model.parameters(), lr=1e-3, weight_decay=0)

---

## C4) Train-only normalization computation (critical detail)

Normalization is computed ONLY from the training split using `compute_norm_stats(dataset, indices)`.

Keys normalized:

* node
* linker
* base

For each training sample:

* accumulate sum and squared sum of each feature vector
* compute mean = sum/n
* compute variance = (sq/n) - mean^2
* std = sqrt(clamp(var, 1e-12))

This prevents:

* std = 0 issues
* leakage of statistics from val/test into training preprocessing

Workflow:

1. build `raw = COFModel1Dataset(...)` with no norm_stats
2. split indices
3. compute `norm_stats` from raw over training indices
4. rebuild final dataset = COFModel1Dataset(..., norm_stats)

So val/test samples are normalized using training-set mean/std, which is correct.

---

## C5) Split strategies (three modes)

`SPLIT_MODE` controls the meaning of generalization.

### Mode 1: strat_topology (default)

Uses `split_stratified_by_topology_safe(topo_labels)`.

Key behavior per topology_id:

* if a topology has 1 sample → goes to TRAIN
* if 2 samples → 1 TRAIN, 1 TEMP
* if >=3 samples:

  * allocate approx `test_size=0.30` to TEMP, remainder TRAIN

Then TEMP is shuffled and split into:

* TEST fraction = `val_size_of_temp=0.50` of TEMP
* VAL = rest

So overall:

* train ~70%
* val ~15%
* test ~15%
  but adjusted per topology safely.

This is designed to avoid failures from rare topologies.

### Mode 2: holdout_parent2c

Uses `split_holdout_by_group(groups=parent_2c_id)`.

This ensures:

* parent_2c groups do not overlap between train/val/test

Meaning: evaluates generalization to unseen 2C parent linkers.

### Mode 3: holdout_nodes

Uses `split_holdout_by_node_ids(node1_ids, node2_ids)`.

Process:

1. collect all unique node linker IDs from node1 and node2 (excluding -1)
2. shuffle
3. select:

   * test_nodes = ~15% of unique node IDs
   * val_nodes  = ~15% next chunk
4. assign a COF:

   * to TEST if it contains any test_node
   * else to VAL if contains any val_node
   * else to TRAIN

This guarantees:

* held-out node IDs do not appear in TRAIN

It also prints a leak check.

---

## C6) Split sanity checks (printed)

After splitting, `train.py` prints:

* mode
* total N
* train/val/test sizes
* overlaps between splits (should be 0)
* unique group counts depending on split mode
* leak check for holdout_nodes (should be 0)

This makes the splitting auditable for reviewers.

---

## C7) DataLoaders

Constructed as:

* train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
* val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)
* test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)

No custom collate function is used because each sample already returns fixed-size vectors.

---

## C8) Input dimension discovery (automatic)

The model’s input dimensions are inferred from the first dataset sample:

* `sample = dataset[0]`
* `dims = {k: v.shape[0] for k,v in sample.items() if k != "y"}`

So dims are computed for:

* topo, node, linker, base, misc

This ensures the model is compatible even if descriptor CSV feature counts change (as long as metadata is consistent).

---

## C9) Training loop (exact behavior)

Epoch loop runs up to 2000 epochs:

### Training phase:

* model.train()
* iterate batches b:

  * optimizer.zero_grad()
  * pred = model(b)
  * loss = MSE(pred, b['y'])
  * loss.backward()
  * optimizer.step()

### Validation phase:

* model.eval()
* compute val MSE across val_loader without gradients
* print:

  * `Epoch XXXX | Val MSE = ...`

### Early stopping:

* if val MSE improves:

  * store `best_state = model.state_dict()` (moved to CPU)
  * reset patience counter
* else:

  * increment patience counter
  * if patience reaches 25:

    * stop training

So the model stored is the best by validation MSE.

---

## C10) Test evaluation (after restoring best weights)

After training:

* load `best_state`

* run inference over test_loader

* compute metrics separately for each output dimension:

* MAE LCD

* MAE PLD

* R2 LCD

* R2 PLD

These are printed.

---

## C11) Saved artifact: model1_final.pt

The final saved file is:

* `model1_final.pt`

It contains a full dictionary:

* model_state_dict: best weights
* norm_stats: mean/std used for normalization
* dims: input dimensions for reconstruction
* hyperparameters: lr, batch_size, weight_decay, seed
* split_mode: which split strategy was used
* metrics: MAE and R2 values
* y_true: full test targets array
* y_pred: full test predictions array

This makes the model reproducible and portable for future inference.

---

# 3) What to call this model in thesis / journal

Use this exact naming (accurate and reviewer-friendly):

**COFModel1: Multi-branch MLP (late-fusion) deep regressor for simultaneous prediction of LCD and PLD**

And describe it in one line:

* “A PyTorch multi-input neural network that encodes topology embeddings, linker descriptors, base descriptors, and functionalization metadata using separate MLP encoders, then fuses them to predict LCD and PLD.”

---

# 4) Practical “How to run training” (external user)

From repo root, ensure these exist:

* `cof_meta_model1_clean.csv`
* `embed/` directory with `.pt` embeddings referenced by `topology_pt_path`
* `features/` directory with descriptor CSVs

Then run:

```
python train.py
```

Output:

* console logs (validation MSE per epoch)
* `model1_final.pt`

---

# 5) Notes / Assumptions / Common failure points

1. `bridge_type` must be one of: `none, ch2, ph, dir`
2. `topology_pt_path` must exist and point to valid `.pt` objects
3. Descriptor CSVs must contain `linker_id` and numeric columns
4. If your descriptor tables change column prefixes, feature-family masking may not work (it relies on `geom_` and `RDF_`)
5. `edge_fn_id` and `base_id` are allowed to be `-1`, in which case the dataset uses zero vectors
6. Topology embeddings are not normalized; if their scale changes drastically between runs, that can impact performance

---

