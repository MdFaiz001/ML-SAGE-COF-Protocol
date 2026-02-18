import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from dataset import COFModel1Dataset
from model import COFModel1


# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ============================================================
# PATHS (KEEP SAME AS YOUR WORKING PROTOCOL)
# ============================================================
META = "cof_meta_model1_clean.csv"
EMBED = "embed"

FEATURES = {
    "2c_fn": "features/2_con_fn_linker_feat_ms.csv",
    "2c_unfn": "features/2_con_unfn_linker_feat_ms.csv",
    "3c": "features/3_con_linker_feat_ms.csv",
    "4c": "features/4_con_linker_feat_ms.csv",
    "base": "features/base_feat_ms.csv",
}


# ============================================================
# TRAINING HYPERPARAMETERS (KEEP SAME)
# ============================================================
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 2000
PATIENCE = 25


# ============================================================
# SPLIT MODE (THIS DETERMINES WHICH MODE RUNS)
# ============================================================
# Choose ONE:
#   "strat_topology"   -> stratified random by topology (SAFE for singleton topologies)
#   "holdout_parent2c" -> strict heldout by parent_2c_id (unseen 2C parents)
#   "holdout_nodes"    -> strict heldout by node linker IDs (unseen 3C/4C nodes)
SPLIT_MODE = "strat_topology"
RANDOM_STATE = 42


# ============================================================
# NORMALIZATION (TRAIN SET ONLY) (KEEP SAME LOGIC)
# ============================================================
def compute_norm_stats(dataset, indices):
    keys = ["node", "linker", "base"]
    stats = {k: {"sum": 0, "sq": 0} for k in keys}
    n = 0

    for i in indices:
        s = dataset[int(i)]
        for k in keys:
            stats[k]["sum"] += s[k]
            stats[k]["sq"] += s[k] ** 2
        n += 1

    out = {}
    for k in keys:
        mean = stats[k]["sum"] / n
        var = stats[k]["sq"] / n - mean ** 2
        out[k] = {
            "mean": mean,
            "std": torch.sqrt(torch.clamp(var, 1e-12)),
        }
    return out


# ============================================================
# SPLIT HELPERS
# ============================================================
def split_stratified_by_topology_safe(topo_labels, test_size=0.30, val_size_of_temp=0.50, seed=42):
    """
    Safe stratified split by topology_id that works even if some topologies have only 1 sample.

    Per topology:
      n=1  -> TRAIN
      n=2  -> 1 TRAIN, 1 TEMP
      n>=3 -> allocate approx test_size to TEMP, rest TRAIN
    Then TEMP is split into TEST and VAL (val_size_of_temp fraction goes to TEST).
    """
    rng = np.random.RandomState(seed)
    topo_labels = np.asarray(topo_labels)

    train_idx = []
    temp_idx = []

    unique_topos = np.unique(topo_labels)
    for t in unique_topos:
        idxs = np.where(topo_labels == t)[0]
        rng.shuffle(idxs)
        n = len(idxs)

        if n == 1:
            train_idx.extend(idxs.tolist())
        elif n == 2:
            train_idx.append(int(idxs[0]))
            temp_idx.append(int(idxs[1]))
        else:
            n_temp = max(1, int(round(test_size * n)))
            n_train = n - n_temp
            if n_train < 1:
                n_train = 1
                n_temp = n - 1
            train_idx.extend(idxs[:n_train].tolist())
            temp_idx.extend(idxs[n_train:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    temp_idx = np.array(temp_idx, dtype=int)

    # Split TEMP -> TEST/VAL
    rng.shuffle(temp_idx)
    n_temp = len(temp_idx)
    if n_temp == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    n_test = max(1, int(round(val_size_of_temp * n_temp)))
    test_idx = temp_idx[:n_test]
    val_idx = temp_idx[n_test:]
    return train_idx, val_idx, test_idx


def split_holdout_by_group(groups, test_size=0.30, val_size_of_temp=0.50, seed=42):
    """
    Strict held-out split by group (no group overlap).
    Correctly maps relative indices -> global indices.
    """
    all_idx = np.arange(len(groups))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(all_idx, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_of_temp, random_state=seed)
    val_rel, test_rel = next(gss2.split(np.arange(len(temp_idx)), groups=groups[temp_idx]))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def split_holdout_by_node_ids(node1_ids, node2_ids, test_frac=0.15, val_frac=0.15, seed=42):
    """
    Hold out node-linker IDs (3C/4C):
      - pick node IDs for test and val
      - any COF containing test node IDs -> test
      - else containing val node IDs -> val
      - rest -> train
    Guarantees no held-out node IDs appear in training.
    """
    rng = np.random.RandomState(seed)

    all_nodes = np.unique(np.concatenate([node1_ids, node2_ids]))
    all_nodes = all_nodes[all_nodes != -1]
    rng.shuffle(all_nodes)

    n_total = len(all_nodes)
    n_test = max(1, int(round(test_frac * n_total)))
    n_val = max(1, int(round(val_frac * n_total)))

    test_nodes = set(all_nodes[:n_test].tolist())
    val_nodes = set(all_nodes[n_test:n_test + n_val].tolist())

    test_mask = np.array([(a in test_nodes) or (b in test_nodes) for a, b in zip(node1_ids, node2_ids)])
    val_mask = np.array([(a in val_nodes) or (b in val_nodes) for a, b in zip(node1_ids, node2_ids)])

    # priority: test > val > train
    val_mask = val_mask & (~test_mask)
    train_mask = (~test_mask) & (~val_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    return train_idx, val_idx, test_idx, test_nodes, val_nodes


# ============================================================
# BUILD RAW DATASET + SPLIT
# ============================================================
raw = COFModel1Dataset(META, FEATURES, EMBED)

N = len(raw)
topo = raw.meta["topology_id"].values

# required for holdout modes (must exist in meta CSV)
parent2c = raw.meta["parent_2c_id"].values
node1 = raw.meta["node1_linker_id"].values
node2 = raw.meta["node2_linker_id"].values

if SPLIT_MODE == "strat_topology":
    train_idx, val_idx, test_idx = split_stratified_by_topology_safe(
        topo, test_size=0.30, val_size_of_temp=0.50, seed=RANDOM_STATE
    )

elif SPLIT_MODE == "holdout_parent2c":
    train_idx, val_idx, test_idx = split_holdout_by_group(
        parent2c, test_size=0.30, val_size_of_temp=0.50, seed=RANDOM_STATE
    )

elif SPLIT_MODE == "holdout_nodes":
    train_idx, val_idx, test_idx, test_nodes, val_nodes = split_holdout_by_node_ids(
        node1, node2, test_frac=0.15, val_frac=0.15, seed=RANDOM_STATE
    )

else:
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")


# ============================================================
# SPLIT SANITY CHECK
# ============================================================
print("\n===== SPLIT SANITY CHECK =====")
print("Mode:", SPLIT_MODE)
print("Total N:", N)
print("Train/Val/Test:", len(train_idx), len(val_idx), len(test_idx))
print("Overlap train∩val:", len(set(train_idx).intersection(set(val_idx))))
print("Overlap train∩test:", len(set(train_idx).intersection(set(test_idx))))
print("Overlap val∩test:", len(set(val_idx).intersection(set(test_idx))))

if SPLIT_MODE == "strat_topology":
    print("Unique topologies train/val/test:",
          len(set(topo[train_idx])),
          len(set(topo[val_idx])),
          len(set(topo[test_idx])))

elif SPLIT_MODE == "holdout_parent2c":
    print("Unique parent_2c train/val/test:",
          len(set(parent2c[train_idx])),
          len(set(parent2c[val_idx])),
          len(set(parent2c[test_idx])))

elif SPLIT_MODE == "holdout_nodes":
    train_nodes = set(np.unique(np.concatenate([node1[train_idx], node2[train_idx]])).tolist())
    leak_test = len(set(test_nodes).intersection(train_nodes))
    leak_val = len(set(val_nodes).intersection(train_nodes))
    print("Held-out node IDs | test:", len(test_nodes), " val:", len(val_nodes))
    print("Leak check (should be 0) | test_nodes∩train_nodes:", leak_test,
          " val_nodes∩train_nodes:", leak_val)

print("==============================\n")


# ============================================================
# FINAL DATASET WITH TRAIN NORMALIZATION
# ============================================================
norm_stats = compute_norm_stats(raw, train_idx)
dataset = COFModel1Dataset(META, FEATURES, EMBED, norm_stats)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)


# ============================================================
# MODEL INIT (KEEP SAME)
# ============================================================
sample = dataset[0]
dims = {k: v.shape[0] for k, v in sample.items() if k != "y"}

model = COFModel1(dims)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.MSELoss()


# ============================================================
# TRAINING LOOP (KEEP SAME)
# ============================================================
best_val = float("inf")
best_state = None
pat = 0

for ep in range(MAX_EPOCHS):

    model.train()
    for b in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(b), b["y"])
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for b in val_loader:
            val_losses.append(loss_fn(model(b), b["y"]).item())

    val_mse = float(np.mean(val_losses))
    print(f"Epoch {ep:04d} | Val MSE = {val_mse:.4f}")

    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping triggered")
            break


# ============================================================
# TEST EVALUATION (KEEP SAME)
# ============================================================
model.load_state_dict(best_state)

y_true, y_pred = [], []
with torch.no_grad():
    for b in test_loader:
        y_true.append(b["y"].numpy())
        y_pred.append(model(b).numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

mae_lcd = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_pld = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
r2_lcd = r2_score(y_true[:, 0], y_pred[:, 0])
r2_pld = r2_score(y_true[:, 1], y_pred[:, 1])

print("\n===== FINAL TEST METRICS =====")
print(f"MAE LCD : {mae_lcd:.3f} Å")
print(f"MAE PLD : {mae_pld:.3f} Å")
print(f"R2  LCD : {r2_lcd:.3f}")
print(f"R2  PLD : {r2_pld:.3f}")


# ============================================================
# SAVE MODEL (KEEP SAME NAME)
# ============================================================
torch.save(
    {
        "model_state_dict": best_state,
        "norm_stats": norm_stats,
        "dims": dims,
        "hyperparameters": {
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "seed": 42,
        },
        "split_mode": SPLIT_MODE,
        "metrics": {
            "MAE_LCD": float(mae_lcd),
            "MAE_PLD": float(mae_pld),
            "R2_LCD": float(r2_lcd),
            "R2_PLD": float(r2_pld),
        },
        "y_true": y_true,
        "y_pred": y_pred,
    },
    "model1_final.pt",
)

print("\n✅ Final Model-1 saved as model1_final.pt")

