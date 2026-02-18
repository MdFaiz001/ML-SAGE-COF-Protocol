#!/usr/bin/env python3
import os
import math
import random
import json
import re
import signal
import multiprocessing as mp
from functools import partial

import pandas as pd
import pormake as pm


# ============================================================
#                USER SETTINGS
# ============================================================

# Number of parallel worker processes
NUM_PROCESSES = 44   # set this based on machine capability (e.g., 8, 12, 16, 24)

# XYZ directories
BASE_DIR_2C   = "./2c_xyz"
BASE_DIR_3C   = "./3c_xyz"
BASE_DIR_4C   = "./4c_xyz"
BASE_DIR_FN2C = "./fn2c_xyz"

# Output
OUTPUT_ROOT = "./generated_cofs_new1"

# CSVs
PLAN_CSV         = "COF_generation_plan.csv"
TOPO_SORTED_CSV  = "topo_sorted.csv"
LINKER_2C_CSV    = "2c_linkers.csv"
LINKER_3C_CSV    = "3c_linkers.csv"
LINKER_4C_CSV    = "4c_linkers.csv"
FN_2C_CSV        = "functionalized_2c_linkers.csv"

# RMSD threshold
RMSD_THRESHOLD = 0.30

# Random seed
RNG_MASTER_SEED = 2025

# Timeout (seconds) for each COF (build + CIF write)
TIME_LIMIT = 180

# Resume logic
START_FROM_COF_ID = None   # or "COF_004501"


# ============================================================
#            TIMEOUT HANDLING (PER WORKER)
# ============================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out.")


# ============================================================
#            TOPOLOGY / PARSERS (unchanged)
# ============================================================

def parse_node_info(node_info: str):
    if not isinstance(node_info, str):
        return {}

    node_info = node_info.replace("–", "-")
    parts = [p.strip() for p in node_info.split(";") if p.strip()]

    node_types = {}

    pattern = re.compile(r"type\s+(\d+)\s*\(CN=(\d+),\s*slots\s+(\d+)(?:-(\d+))?\)")

    for p in parts:
        m = pattern.search(p)
        if not m:
            continue

        idx = int(m.group(1))
        cn  = int(m.group(2))
        s1  = int(m.group(3))
        s2  = m.group(4)

        if s2 is not None:
            s2 = int(s2)
            slots = list(range(s1, s2 + 1))
        else:
            slots = [s1]

        node_types[idx] = {"cn": cn, "slots": slots}
    return node_types


def parse_edge_info(edge_info: str):
    if not isinstance(edge_info, str):
        return {}

    edge_info = edge_info.replace("–", "-")
    parts = [p.strip() for p in edge_info.split(";")]

    edge_types = {}
    pattern = re.compile(r"\((\d+)\s*,\s*(\d+)\)\s*\(slots\s+(\d+)-(\d+)\)")

    for p in parts:
        if not p:
            continue
        m = pattern.search(p)
        if m:
            i     = int(m.group(1))
            j     = int(m.group(2))
            start = int(m.group(3))
            end   = int(m.group(4))
            slots = list(range(start, end + 1))
            edge_types[(i, j)] = slots

    return edge_types


def get_topology_info(topology_name, df_topo):
    row = df_topo.loc[df_topo["Topology"] == topology_name]
    if row.empty:
        raise ValueError(f"Topology {topology_name} not found!")

    row = row.iloc[0]
    return parse_node_info(row["Node info"]), parse_edge_info(row["Edge info"])


# ============================================================
#            BB LOADING FUNCTIONS (unchanged)
# ============================================================

def node_type_label_to_cn(label):
    if not isinstance(label, str):
        return None
    s = label.strip()
    if s.endswith("C"):
        try:
            return int(s[:-1])
        except:
            return None
    return None


def build_node_bbs_for_row(row, node_types, df_3c, df_4c):
    node_bbs = {}

    # node1 → type0
    cn1 = node_type_label_to_cn(row["node1_type"])
    if cn1 is not None:
        linker1 = row["node1_linker"]
        linker_id1 = int(linker1)

        if cn1 == 3:
            rec = df_3c.loc[linker_id1]
            xyz_path = os.path.join(BASE_DIR_3C, rec["xyz_file"])
        else:
            rec = df_4c.loc[linker_id1]
            xyz_path = os.path.join(BASE_DIR_4C, rec["xyz_file"])

        node_bbs[0] = pm.BuildingBlock(xyz_path)

    # node2 → type1 (only when needed)
    if int(row["num_node_types"]) == 2:
        cn2 = node_type_label_to_cn(row["node2_type"])
        if cn2 is not None and pd.notna(row["node2_linker"]):
            linker_id2 = int(row["node2_linker"])

            if cn2 == 3:
                rec = df_3c.loc[linker_id2]
                xyz_path = os.path.join(BASE_DIR_3C, rec["xyz_file"])
            else:
                rec = df_4c.loc[linker_id2]
                xyz_path = os.path.join(BASE_DIR_4C, rec["xyz_file"])

            node_bbs[1] = pm.BuildingBlock(xyz_path)

    return node_bbs


def get_edge_parent_and_fn_bb(row, df_2c, df_fn2c):
    parent_rec = df_2c.loc[row["parent_2c"]]
    parent_bb  = pm.BuildingBlock(os.path.join(BASE_DIR_2C, parent_rec["xyz_file"]))

    fn_name = row["edge_fn_name"]

    if fn_name in (None, "", float("nan")):
        return parent_bb, None

    if fn_name not in df_fn2c.index:
        return parent_bb, None

    fn_rec = df_fn2c.loc[fn_name]
    fn_bb  = pm.BuildingBlock(os.path.join(BASE_DIR_FN2C, fn_rec["xyz_file"]))
    return parent_bb, fn_bb


# ============================================================
#         FUNCTIONALIZATION DISTRIBUTION (Option A)
# ============================================================

def select_functionalized_slots(edge_types, coverage_pct, rng):
    all_slots = []
    for slots in edge_types.values():
        all_slots.extend(slots)
    all_slots = sorted(set(all_slots))
    total_slots = len(all_slots)

    fn_slots = set()
    fn_counts = {etype: 0 for etype in edge_types}

    if total_slots == 0 or coverage_pct <= 0:
        return fn_slots, fn_counts

    target = max(1, round(total_slots * coverage_pct / 100))
    remaining = target

    # At least 1 per edge type
    for etype, slots in edge_types.items():
        if remaining <= 0:
            break
        if not slots:
            continue
        c = rng.choice(slots)
        fn_slots.add(c)
        fn_counts[etype] += 1
        remaining -= 1

    # Fill remainder randomly
    if remaining > 0:
        pool = [s for s in all_slots if s not in fn_slots]
        if pool:
            extra = rng.sample(pool, min(remaining, len(pool)))
            for c in extra:
                fn_slots.add(c)
                for etype, slots in edge_types.items():
                    if c in slots:
                        fn_counts[etype] += 1
                        break

    return fn_slots, fn_counts


# ============================================================
#        WORKER FUNCTION (single COF processing)
# ============================================================

def process_one_cof(row_dict, df_topo, df_2c, df_3c, df_4c, df_fn2c):
    """
    Each worker:
    - Has its own timeout
    - Has its own PORMAKE objects
    - Returns logs (rmsd + coverage)
    """

    signal.signal(signal.SIGALRM, timeout_handler)

    row = row_dict
    cof_id = row["cof_id"]
    topo_name = row["topology_name"]

    output_dir = os.path.join(OUTPUT_ROOT, row["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    coverage_pct = float(row["coverage_pct"])

    # Recreate PORMAKE objects inside each worker (safe)
    database = pm.Database()
    builder  = pm.Builder()
    locator  = pm.Locator()

    # Load topology info
    node_info, edge_info = get_topology_info(topo_name, df_topo)
    topo = database.get_topo(topo_name)

    # Load node building blocks
    try:
        node_bbs = build_node_bbs_for_row(row, node_info, df_3c, df_4c)
    except Exception as e:
        return {
            "cof_id": cof_id,
            "rmsd": [],
            "coverage": {"timed_out": True, "error": str(e)}
        }

    # Load edge building blocks
    parent_bb, fn_bb = get_edge_parent_and_fn_bb(row, df_2c, df_fn2c)
    edge_bbs_parent = {etype: parent_bb for etype in edge_info}

    # -------- RMSD evaluation --------
    rmsd_list = []
    for node_i, bb in node_bbs.items():
        local = topo.unique_local_structures[node_i]
        rmsd_val = locator.calculate_rmsd(local, bb)
        feasible = rmsd_val <= RMSD_THRESHOLD

        rmsd_list.append({
            "node_type_idx": node_i,
            "node_CN": node_info[node_i]["cn"],
            "rmsd": rmsd_val,
            "feasible": feasible,
        })

    # -------- Build BB list --------
    try:
        bbs = builder.make_bbs_by_type(topo, node_bbs, edge_bbs_parent)
    except Exception as e:
        return {
            "cof_id": cof_id,
            "rmsd": rmsd_list,
            "coverage": {"timed_out": True, "error": str(e)}
        }

    # -------- Select functionalized slots --------
    rng = random.Random(RNG_MASTER_SEED + hash(cof_id) % (2**31))
    if fn_bb is None:
        fn_slots = set()
        fn_counts = {etype: 0 for etype in edge_info}
    else:
        fn_slots, fn_counts = select_functionalized_slots(edge_info, coverage_pct, rng)

    # Apply fn BBs
    for s in fn_slots:
        if fn_bb is not None:
            bbs[s] = fn_bb.copy()

    cif_path = os.path.join(output_dir, f"{cof_id}.cif")
    timed_out = False

    # -------- Build + CIF write with timeout --------
    try:
        signal.alarm(TIME_LIMIT)
        MOF = builder.build(topo, bbs)
        MOF.write_cif(cif_path)
    except TimeoutException:
        timed_out = True
    except Exception as e:
        timed_out = True
    finally:
        signal.alarm(0)

    total_edge_slots = sum(len(v) for v in edge_info.values())

    return {
        "cof_id": cof_id,
        "rmsd": rmsd_list,
        "coverage": {
            "timed_out": timed_out,
            "parent_2c": row["parent_2c"],
            "edge_fn_name": row["edge_fn_name"],
            "coverage_pct_target": coverage_pct,
            "total_edge_slots": total_edge_slots,
            "fn_edges_total": len(fn_slots),
            "fn_edges_per_type": {str(k): v for k, v in fn_counts.items()},
        }
    }


# ============================================================
#                MAIN MULTIPROCESS CONTROLLER
# ============================================================

def main():

    # Load CSVs in main process
    df_plan = pd.read_csv(PLAN_CSV)
    df_topo = pd.read_csv(TOPO_SORTED_CSV)
    df_2c   = pd.read_csv(LINKER_2C_CSV).set_index("name")
    df_3c   = pd.read_csv(LINKER_3C_CSV).set_index("name")
    df_4c   = pd.read_csv(LINKER_4C_CSV).set_index("name")
    df_fn2c = pd.read_csv(FN_2C_CSV).set_index("name")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Convert plan rows to dicts
    plan_rows = df_plan.to_dict("records")

    # Resume logic
    if START_FROM_COF_ID is not None:
        skip = True
        filtered = []
        for row in plan_rows:
            if skip:
                if row["cof_id"] == START_FROM_COF_ID:
                    skip = False
                    filtered.append(row)
            else:
                filtered.append(row)
        plan_rows = filtered
        print(f"[INFO] Resuming from {START_FROM_COF_ID}, remaining: {len(plan_rows)}")

    # Prepare worker function with bound arguments
    worker_func = partial(
        process_one_cof,
        df_topo=df_topo,
        df_2c=df_2c,
        df_3c=df_3c,
        df_4c=df_4c,
        df_fn2c=df_fn2c,
    )

    print(f"[INFO] Starting multiprocessing with {NUM_PROCESSES} workers")

    # =====================================================
    #               RUN MULTIPROCESSING
    # =====================================================
    results = []
    with mp.Pool(NUM_PROCESSES) as pool:
        for res in pool.imap_unordered(worker_func, plan_rows):
            results.append(res)
            print(f"[DONE] {res['cof_id']}")

    print(f"[INFO] All COFs processed.")

    # =====================================================
    #               COLLECT AND WRITE LOGS
    # =====================================================

    rmsd_out = []
    cov_out  = []

    for item in results:
        cof_id = item["cof_id"]

        # RMSD
        for rr in item["rmsd"]:
            rmsd_out.append({
                "cof_id": cof_id,
                **rr
            })

        # Coverage
        cov_data = item["coverage"]
        cov_row = {
            "cof_id": cof_id,
            "timed_out": cov_data["timed_out"],
        }
        if not cov_data["timed_out"]:
            cov_row.update({
                "parent_2c": cov_data["parent_2c"],
                "edge_fn_name": cov_data["edge_fn_name"],
                "coverage_pct_target": cov_data["coverage_pct_target"],
                "total_edge_slots": cov_data["total_edge_slots"],
                "fn_edges_total": cov_data["fn_edges_total"],
                "fn_edges_per_type": json.dumps(cov_data["fn_edges_per_type"]),
            })
        cov_out.append(cov_row)

    # Save logs
    pd.DataFrame(rmsd_out).to_csv(os.path.join(OUTPUT_ROOT, "rmsd_log.csv"), index=False)
    pd.DataFrame(cov_out).to_csv(os.path.join(OUTPUT_ROOT, "coverage_log.csv"), index=False)

    print("[INFO] All logs saved.")


if __name__ == "__main__":
    main()

