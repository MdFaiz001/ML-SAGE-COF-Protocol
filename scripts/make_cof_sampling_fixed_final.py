#!/usr/bin/env python3
"""
Generate COF_generation_plan.csv with ~15,000 COF designs
for COF-ML Model 1, with CORRECT CN-based node/linker
assignment for 3–4 topologies.

Uses:
  - topo_sorted.csv
  - 3c_linkers.csv
  - 4c_linkers.csv
  - 2c_linkers.csv
  - functionalized_2c_linkers.csv

Each row of the output CSV describes one COF design:
  - topology_name
  - case_id (1..5)
  - node1_type / node2_type (3C/4C)
  - node1_linker / node2_linker (chosen consistently with CN)
  - parent_2c (unfunctionalized linker)
  - edge_fn_name (functionalized variant, if coverage > 0)
  - coverage_pct (0, 25, 50, 75, 100)
  - bridge_type, base_id, fn_sites (parsed from functionalized name)
  - output_dir
"""

import pandas as pd
import numpy as np
import re
import random
from pathlib import Path

# ---------------- CONFIGURATION ---------------- #

RANDOM_SEED = 12345  # for reproducibility
OUTPUT_CSV = "COF_generation_plan.csv"

# Target number of COFs per case (must sum to 15000)
CASE_TARGETS = {
    1: 6000,  # 1 node, CN=3
    2: 12000,  # 1 node, CN=4
    3: 12000,  # 2 nodes, 3-3
    4: 20000,  # 2 nodes, 4-4
    5: 10000,  # 2 nodes, 3-4 (CN ordering taken from topology)
}

COVERAGES = [0, 25, 50, 75, 100]  # degrees of functionalization (%)

# ---------------- UTILS ---------------- #

def parse_cn_map(node_info: str):
    """
    Parse Node info strings like:
      'type 0 (CN=3, slots 0–1); type 1 (CN=4, slots 2–4)'
    into a dict:
      {0: 3, 1: 4}

    We rely on 'type <idx> (CN=<cn>, ...' pattern.
    """
    if not isinstance(node_info, str):
        return {}

    # normalize en dash
    s = node_info.replace("–", "-")

    pattern = re.compile(r"type\s+(\d+)\s*\(CN\s*=\s*(\d+)", re.IGNORECASE)
    cn_map = {}
    for m in pattern.finditer(s):
        idx = int(m.group(1))
        cn = int(m.group(2))
        cn_map[idx] = cn
    return cn_map


def classify_case(num_node_types: int, cns):
    """
    Classify topology into one of the 5 cases:
      1: 1 node, CN=3
      2: 1 node, CN=4
      3: 2 nodes, 3-3
      4: 2 nodes, 4-4
      5: 2 nodes, 3-4 (in any order)

    Returns case_id (1..5) or None if it doesn't fit.
    """
    if num_node_types == 1 and len(cns) == 1:
        if cns[0] == 3:
            return 1
        elif cns[0] == 4:
            return 2
        else:
            return None

    if num_node_types == 2 and len(cns) == 2:
        s = set(cns)
        if s == {3}:
            return 3
        elif s == {4}:
            return 4
        elif s == {3, 4}:
            return 5
        else:
            return None

    # Fallback
    return None


def assign_quota_per_topology(df_case, target_total):
    """
    Given a subset of topologies for a case and a target total,
    assign an integer quota per topology that sums to target_total.

    Strategy: base = target // n; remainder distributed by +1 to
    the first 'remainder' topologies after shuffling.
    """
    n = len(df_case)
    if n == 0 or target_total == 0:
        return {idx: 0 for idx in df_case.index}

    base = target_total // n
    remainder = target_total % n

    indices = list(df_case.index)
    random.shuffle(indices)

    quota = {idx: base for idx in indices}
    for i in range(remainder):
        quota[indices[i]] += 1

    return quota


def parse_functionalized_name(name: str):
    """
    Parse a functionalized 2C linker name like:
      dir_1_c_link__2_NMe2__site1
      ch2_5_n_link__11_pyrazine__site2
      ph_9_n_link__17__bsite1__lsite3

    Return dict with keys:
      bridge_type, parent_2c, base_id, sites
    """
    parts = name.split("__")
    if len(parts) < 2:
        return {"bridge_type": None, "parent_2c": None,
                "base_id": None, "sites": None}

    prefix = parts[0]          # e.g. "dir_1_c_link"
    rest = parts[1:]           # e.g. ["2_NMe2", "site1"] or ["17","bsite1","lsite3"]

    # bridge_type + parent_2c
    if "_" in prefix:
        bridge_type, parent_2c = prefix.split("_", 1)
    else:
        bridge_type, parent_2c = None, prefix

    # base_id and sites
    if len(rest) == 1:
        base_id = rest[0]
        sites = None
    elif len(rest) == 2:
        base_id = rest[0]
        sites = rest[1]
    else:
        base_id = rest[0]
        sites = "__".join(rest[1:])

    return {
        "bridge_type": bridge_type,
        "parent_2c": parent_2c,
        "base_id": base_id,
        "sites": sites,
    }


# ---------------- MAIN SAMPLING LOGIC ---------------- #

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load basic CSVs --- #
    topo_df = pd.read_csv("topo_sorted.csv")
    c3_df = pd.read_csv("3c_linkers.csv")
    c4_df = pd.read_csv("4c_linkers.csv")
    c2_un_df = pd.read_csv("2c_linkers.csv")
    fn2_df = pd.read_csv("functionalized_2c_linkers.csv")

    # Column name assumptions (adjust here if needed)
    topo_name_col = "Topology"
    num_node_types_col = "# Node types"
    node_info_col = "Node info"

    c3_name_col = "name"
    c4_name_col = "name"
    c2_un_name_col = "name"
    fn2_name_col = "name"

    # --- Node linker lists --- #
    c3_names = list(c3_df[c3_name_col].unique())  # 3-connected linkers
    c4_names = list(c4_df[c4_name_col].unique())  # 4-connected linkers

    # --- Parent 2C linkers --- #
    parent_2c_names = list(c2_un_df[c2_un_name_col].unique())

    # --- Process functionalized 2C linkers --- #
    fn2_df = fn2_df.copy()
    parsed = fn2_df[fn2_name_col].apply(parse_functionalized_name)
    fn2_df["bridge_type"] = parsed.apply(lambda d: d["bridge_type"])
    fn2_df["parent_2c"] = parsed.apply(lambda d: d["parent_2c"])
    fn2_df["base_id"] = parsed.apply(lambda d: d["base_id"])
    fn2_df["sites"] = parsed.apply(lambda d: d["sites"])

    # Build mapping: parent_2c -> list of functionalized variants (rows)
    parent_to_fn_variants = {}
    for parent in parent_2c_names:
        sub = fn2_df[fn2_df["parent_2c"] == parent]
        parent_to_fn_variants[parent] = sub.reset_index(drop=True)

    # --- Parse CN maps from Node info --- #
    topo_df = topo_df.copy()
    topo_df["cn_map"] = topo_df[node_info_col].apply(parse_cn_map)
    # Ordered CN list (type 0, type 1, ...) for classification
    topo_df["cns"] = topo_df["cn_map"].apply(
        lambda d: [d[k] for k in sorted(d.keys())] if isinstance(d, dict) and d else []
    )

    # --- Classify topologies into cases --- #
    topo_df["case_id"] = topo_df.apply(
        lambda row: classify_case(
            int(row[num_node_types_col]),
            row["cns"],
        ),
        axis=1,
    )

    # Filter only valid cases 1..5
    valid_cases = topo_df["case_id"].isin([1, 2, 3, 4, 5])
    topo_df = topo_df[valid_cases].reset_index(drop=True)

    # Group by case
    case_to_topos = {
        case: topo_df[topo_df["case_id"] == case]
        for case in [1, 2, 3, 4, 5]
    }

    # --- Assign quota per topology within each case --- #
    topo_quota = {}  # key: topo_df index, value: quota
    for case_id, df_case in case_to_topos.items():
        target = CASE_TARGETS.get(case_id, 0)
        quotas = assign_quota_per_topology(df_case, target)
        topo_quota.update(quotas)

    # Sanity check: total count
    total_samples = sum(topo_quota.values())
    print(f"[info] Total planned COFs = {total_samples} (target ~15000)")

    # --- Sampling loop --- #
    rows = []
    cof_counter = 0

    for idx, topo_row in topo_df.iterrows():
        case_id = int(topo_row["case_id"])
        n_samples = topo_quota.get(idx, 0)
        if n_samples <= 0:
            continue

        topology_name = str(topo_row[topo_name_col])
        num_nodes = int(topo_row[num_node_types_col])
        cn_map = topo_row["cn_map"]  # dict: type_index -> CN

        for _ in range(n_samples):
            cof_counter += 1
            cof_id = f"COF_{cof_counter:06d}"

            # --- Node linker sampling based on case + CN map --- #
            node1_type = None
            node2_type = None
            node1_linker = None
            node2_linker = None

            if case_id in (1, 2):
                # 1 node type cases
                # case 1 -> CN=3; case 2 -> CN=4
                if case_id == 1:
                    node1_type = "3C"
                    node1_linker = random.choice(c3_names)
                else:  # case_id == 2
                    node1_type = "4C"
                    node1_linker = random.choice(c4_names)

            elif case_id in (3, 4):
                # 2 nodes but both CNs identical: 3-3 or 4-4
                # We can safely assign both according to case.
                if case_id == 3:
                    node1_type = "3C"
                    node2_type = "3C"
                    node1_linker = random.choice(c3_names)
                    node2_linker = random.choice(c3_names)
                else:  # case_id == 4
                    node1_type = "4C"
                    node2_type = "4C"
                    node1_linker = random.choice(c4_names)
                    node2_linker = random.choice(c4_names)

            elif case_id == 5:
                # 2 nodes with CN={3,4}, BUT we must respect the
                # CN ordering from the topology:
                #   node1_type = CN(type 0)
                #   node2_type = CN(type 1)
                # and choose linkers consistently with that.
                if not isinstance(cn_map, dict):
                    # fallback (should not happen)
                    raise ValueError(
                        f"Missing cn_map for 3–4 topology: {topology_name}"
                    )

                cn0 = cn_map.get(0)
                cn1 = cn_map.get(1)

                if cn0 not in (3, 4) or cn1 not in (3, 4):
                    raise ValueError(
                        f"Unexpected CNs in 3–4 case for {topology_name}: {cn_map}"
                    )

                # OPTION A: follow topology ordering
                node1_type = f"{cn0}C"
                node2_type = f"{cn1}C"

                # Choose linkers matching each CN
                if cn0 == 3:
                    node1_linker = random.choice(c3_names)
                else:  # cn0 == 4
                    node1_linker = random.choice(c4_names)

                if cn1 == 3:
                    node2_linker = random.choice(c3_names)
                else:  # cn1 == 4
                    node2_linker = random.choice(c4_names)

            else:
                # Shouldn't happen, but skip just in case
                continue

            # --- Edge (2C) sampling: parent + fn variant + coverage --- #
            parent_2c = random.choice(parent_2c_names)
            coverage_pct = random.choice(COVERAGES)

            edge_fn_name = ""
            bridge_type = ""
            base_id = ""
            sites = ""

            if coverage_pct == 0:
                # unfunctionalized only
                pass
            else:
                variants_df = parent_to_fn_variants.get(parent_2c, None)
                if variants_df is None or len(variants_df) == 0:
                    # fallback: no variants known, degrade to 0% coverage
                    coverage_pct = 0
                else:
                    # Choose a random functionalized variant of this parent
                    vrow = variants_df.sample(
                        1,
                        random_state=random.randint(0, 10**9)
                    ).iloc[0]
                    edge_fn_name = vrow[fn2_name_col]
                    bridge_type = vrow["bridge_type"]
                    base_id = vrow["base_id"]
                    sites = vrow["sites"]

            # Construct an output dir name
            output_dir = f"{topology_name}__{cof_id}"

            rows.append({
                "cof_id": cof_id,
                "topology_name": topology_name,
                "case_id": case_id,
                # node info
                "num_node_types": num_nodes,
                "node1_type": node1_type,
                "node1_linker": node1_linker,
                "node2_type": node2_type,
                "node2_linker": node2_linker,
                # edge info
                "parent_2c": parent_2c,
                "edge_fn_name": edge_fn_name,
                "coverage_pct": coverage_pct,
                "bridge_type": bridge_type,
                "base_id": base_id,
                "fn_sites": sites,
                # misc
                "output_dir": output_dir,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[info] Wrote {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

