\
import pandas as pd
from pathlib import Path
import numpy as np

# ==============================
# USER CONFIG (EDIT THESE PATHS)
# ==============================
FINAL_COF = "final_cof_ml.csv"
GLOBAL_PROPS = "global_props_opt_clean.csv"
TOPO_CSV = "topo_final.csv"

FN_2C = "2_con_fn_linker_feat_ms.csv"
UNFN_2C = "2_con_unfn_linker_feat_ms.csv"
LINKER_3C = "3_con_linker_feat_ms.csv"
LINKER_4C = "4_con_linker_feat_ms.csv"
BASE_CSV = "base_feat_ms.csv"

EMBED_DIR = Path("embed")  # contains {topology_id}.pt
OUT_META = "cof_meta_model1.csv"

# ==============================
# HELPERS
# ==============================
def norm_key(x):
    """Normalize keys for name↔id mapping (handles ints, floats like 4.0, strings, NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Convert things like "4.0" -> "4"
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        s = s.replace(".0", "")
    return s

def parse_connectivity(node_type):
    """Convert '3C'/'4C' -> 3/4 (returns None if empty)."""
    k = norm_key(node_type)
    if k is None:
        return None
    k = k.upper().replace(" ", "")
    if k.endswith("C") and k[:-1].isdigit():
        return int(k[:-1])
    # allow '3'/'4'
    if k.isdigit():
        return int(k)
    raise ValueError(f"Unrecognized node_type: {node_type}")

def build_map(feature_csv, name_col="name_x", id_col="linker_id"):
    df = pd.read_csv(feature_csv)
    if name_col not in df.columns:
        raise KeyError(f"{feature_csv} missing column '{name_col}'. Found: {df.columns.tolist()}")
    if id_col not in df.columns:
        raise KeyError(f"{feature_csv} missing column '{id_col}'. Found: {df.columns.tolist()}")
    keys = df[name_col].apply(norm_key)
    vals = df[id_col].astype(int)
    return dict(zip(keys, vals))

# ==============================
# LOAD DATA
# ==============================
cof = pd.read_csv(FINAL_COF)
props = pd.read_csv(GLOBAL_PROPS)
topo = pd.read_csv(TOPO_CSV)

# Validate expected columns in final_cof_ml.csv
required_cof_cols = [
    "cof_id","topology_name","num_node_types",
    "node1_type","node1_linker","node2_type","node2_linker",
    "parent_2c","edge_fn_name","coverage_pct","bridge_type","base_id"
]
missing = [c for c in required_cof_cols if c not in cof.columns]
if missing:
    raise KeyError(f"final_cof_ml.csv missing columns: {missing}. Found: {cof.columns.tolist()}")

# Topology mapping: topo_final.csv uses 'name' for topology string and 'topology_id' for int id
if "name" not in topo.columns or "topology_id" not in topo.columns:
    raise KeyError(f"topo_final.csv must contain 'name' and 'topology_id'. Found: {topo.columns.tolist()}")

topo_map = dict(zip(topo["name"].apply(norm_key), topo["topology_id"].astype(int)))

# Feature maps (name_x -> linker_id)
fn2c_map   = build_map(FN_2C,   name_col="name_x", id_col="linker_id")
unfn2c_map = build_map(UNFN_2C, name_col="name_x", id_col="linker_id")
l3c_map    = build_map(LINKER_3C, name_col="name_x", id_col="linker_id")
l4c_map    = build_map(LINKER_4C, name_col="name_x", id_col="linker_id")
base_map   = build_map(BASE_CSV, name_col="name_x", id_col="linker_id")

# ==============================
# MERGE TARGETS (LCD/PLD)
# ==============================
props = props[["cof_id","LCD_A","PLD_A"]].copy()
props.rename(columns={"LCD_A":"LCD","PLD_A":"PLD"}, inplace=True)
cof = cof.merge(props, on="cof_id", how="inner")

# ==============================
# BUILD META
# ==============================
rows = []
dropped = {
    "missing_topo_id": 0,
    "missing_pt": 0,
    "missing_parent_2c": 0,
    "missing_edge_fn": 0,
    "missing_node1": 0,
    "missing_node2": 0,
    "missing_base": 0,
    "other": 0
}

for _, r in cof.iterrows():
    try:
        cof_id = r["cof_id"]

        topo_name = norm_key(r["topology_name"])
        topo_id = topo_map.get(topo_name, None)
        if topo_id is None:
            dropped["missing_topo_id"] += 1
            continue

        topo_pt = EMBED_DIR / f"{topo_id}.pt"
        if not topo_pt.exists():
            dropped["missing_pt"] += 1
            continue

        # Node type handling
        ntypes = int(r["num_node_types"])
        node1_conn = parse_connectivity(r["node1_type"])
        node1_name = norm_key(r["node1_linker"])

        if node1_conn == 3:
            node1_id = l3c_map.get(node1_name, None)
        elif node1_conn == 4:
            node1_id = l4c_map.get(node1_name, None)
        else:
            node1_id = None

        if node1_id is None:
            dropped["missing_node1"] += 1
            continue

        # Node2: if only 1 node type, mirror node1
        node2_conn = parse_connectivity(r["node2_type"]) if ntypes == 2 else node1_conn
        node2_name = norm_key(r["node2_linker"]) if ntypes == 2 else node1_name

        if node2_conn == 3:
            node2_id = l3c_map.get(node2_name, None)
        elif node2_conn == 4:
            node2_id = l4c_map.get(node2_name, None)
        else:
            node2_id = None

        if node2_id is None:
            dropped["missing_node2"] += 1
            continue

        # Parent 2C (unfunctionalized)
        parent_2c_name = norm_key(r["parent_2c"])
        parent_2c_id = unfn2c_map.get(parent_2c_name, None)
        if parent_2c_id is None:
            dropped["missing_parent_2c"] += 1
            continue

        # Functionalized edge 2C (optional when coverage=0)
        cov = float(r["coverage_pct"])
        edge_fn_name = norm_key(r["edge_fn_name"])
        if cov == 0 or edge_fn_name is None:
            edge_fn_id = -1
            edge_linker_type = "unfn"
        else:
            edge_fn_id = fn2c_map.get(edge_fn_name, None)
            if edge_fn_id is None:
                dropped["missing_edge_fn"] += 1
                continue
            edge_linker_type = "fn"

        # Base (optional)
        base_name = norm_key(r["base_id"])
        base_id = base_map.get(base_name, -1)

        # Functionalization type inferred (optional)
        if cov == 0 or edge_fn_id == -1:
            func_type = "none"
        else:
            # you can refine later (acid/base/acid_base) if you encode that elsewhere
            func_type = "base" if base_id != -1 else "functionalized"

        rows.append({
            "cof_id": cof_id,

            "topology_name": topo_name,
            "topology_id": topo_id,
            "topology_pt_path": str(topo_pt),

            "node1_connectivity": node1_conn,
            "node1_linker_id": int(node1_id),

            "node2_connectivity": node2_conn,
            "node2_linker_id": int(node2_id),

            "parent_2c_id": int(parent_2c_id),
            "edge_fn_id": int(edge_fn_id),
            "edge_linker_type": edge_linker_type,

            "coverage_pct": cov,
            "coverage_fraction": cov/100.0,
            "bridge_type": norm_key(r["bridge_type"]),
            "base_id": int(base_id),
            "fn_sites": norm_key(r.get("fn_sites", None)),

            "functionalization_type": func_type,

            "LCD": float(r["LCD"]),
            "PLD": float(r["PLD"]),
        })

    except Exception:
        dropped["other"] += 1
        continue

meta = pd.DataFrame(rows).drop_duplicates(subset=["cof_id"])
meta.to_csv(OUT_META, index=False)

print(f"✅ Wrote {OUT_META} with {len(meta)} COFs")
print("Dropped counts:", dropped)
print("Columns:", meta.columns.tolist())
