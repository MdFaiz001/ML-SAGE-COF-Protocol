#!/usr/bin/env python3
import os
import argparse
import random
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_topology_pt(embed_dir: str, topo_id: int) -> str:
    # You told: topology PTs are in embed as topology_id.pt
    # (example: embed/7.pt)
    pt = os.path.join(embed_dir, f"{int(topo_id)}.pt")
    return pt


def compute_expected_edges(edge_slots: int, coverage_fraction: float) -> Tuple[int, int]:
    # Expected number of functionalized vs unfunctionalized edges.
    # We use rounding to closest integer while keeping bounds.
    cov = float(coverage_fraction)
    n_fn = int(round(edge_slots * cov))
    n_fn = max(0, min(edge_slots, n_fn))
    n_un = int(edge_slots - n_fn)
    return n_fn, n_un


# ============================================================
# Parse topology specs (topo_final.csv)
# ============================================================
def _parse_cn_list(node_info: str) -> List[int]:
    """
    Accepts formats like:
      "type 0 (CN=3, slots 0–5)"
      "type 0 (CN=3, slots 0–15) type 1 (CN=4, slots 16–23)"
    Also handles hyphen '-' or en-dash '–' in slots.
    """
    if not isinstance(node_info, str):
        return []
    # Normalize en-dash
    s = node_info.replace("–", "-")
    cns = []
    for part in s.split(")"):
        if "CN=" in part:
            try:
                cn = int(part.split("CN=")[1].split(",")[0].strip())
                cns.append(cn)
            except Exception:
                pass
    return cns


def parse_topology_specs(topo_csv: str) -> Dict[int, Dict]:
    df = pd.read_csv(topo_csv)
    # required columns (as you described)
    req = [
        "topology_id",
        "Total slots",
        "Node slots",
        "Edge slots",
        "# Node types",
        "Node info",
    ]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"topo_final.csv missing required column '{c}'")

    specs: Dict[int, Dict] = {}
    for _, r in df.iterrows():
        topo_id = int(r["topology_id"])
        node_info = r["Node info"]
        cns = _parse_cn_list(node_info)
        # If node types is 1, we still set cn2=cn1 for safety.
        node_types = int(r["# Node types"])
        if len(cns) == 0:
            raise ValueError(
                f"Could not parse node connectivities from Node info for topology_id={topo_id}\n"
                f"Node info: {node_info}\n"
                f"Expected format containing 'CN=' like: type 0 (CN=3, slots 0-5)"
            )
        cn1 = int(cns[0])
        cn2 = int(cns[1]) if (node_types >= 2 and len(cns) >= 2) else int(cn1)

        specs[topo_id] = {
            "total_slots": int(r["Total slots"]),
            "node_slots": int(r["Node slots"]),
            "edge_slots": int(r["Edge slots"]),
            "node_types": int(r["# Node types"]),
            "node1_cn": cn1,
            "node2_cn": cn2,
        }
    return specs


# ============================================================
# Feature pools (IDs available in feature tables)
# ============================================================
def load_id_pool(path_csv: str, id_col: str = "linker_id") -> List[int]:
    df = pd.read_csv(path_csv)
    if id_col not in df.columns:
        raise KeyError(f"{os.path.basename(path_csv)} missing required column '{id_col}'")
    # ensure int IDs
    ids = sorted({int(x) for x in df[id_col].dropna().tolist()})
    return ids


def load_edge_bundle_library(edge_bundle_csv: str, base_id: int) -> pd.DataFrame:
    df = pd.read_csv(edge_bundle_csv)
    req = ["edge_fn_id", "parent_2c_id", "base_id", "bridge_type"]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"edge_bundle_library.csv missing required column '{c}'")

    dfb = df[df["base_id"].astype(int) == int(base_id)].copy()
    if len(dfb) == 0:
        raise ValueError(f"No edge bundles found for base_id={base_id} in {edge_bundle_csv}")
    return dfb


# User said coverage can be 0, 0.25, 0.50, 0.75, 1.0,
# but for base-fixed GA we DO NOT want "no base" genomes, so disallow 0.
ALLOWED_COVERAGE = [0.25, 0.50, 0.75, 1.0]  # coverage=0 disallowed for base-fixed GA


# ============================================================
# Fitness (ranking-based)
# ============================================================
def apply_constraints(df: pd.DataFrame, pld_col: str, av_col: str,
                      pld_min: float, av_min: float) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    ok &= df[pld_col].astype(float) >= float(pld_min)
    ok &= df[av_col].astype(float) >= float(av_min)
    return ok


def compute_rank_fitness(df: pd.DataFrame,
                         pld_col: str,
                         av_col: str,
                         flp_col: str,
                         pld_min: float,
                         av_min: float) -> pd.DataFrame:
    """
    Base-fixed GA fitness (as we discussed):

      1) Model-1 / Model-1b predictions are **hard filters only**
         (PLD >= pld_min AND AV >= av_min).

      2) Among the *valid* genomes, **fitness is driven only by Model-2**
         (higher FLP score is better).

    To keep the GA selection numerically stable and presentation-friendly:
      - fitness is scaled to [0, 1] within each generation for valid genomes
      - invalid genomes get fitness = 0 and are always ranked below valids
        (but we avoid huge negative penalties).

    The raw model-2 score is still written to `fitness_raw` for reporting.
    """
    df = df.copy()

    # hard constraints (filters)
    ok = apply_constraints(df, pld_col, av_col, pld_min, av_min)
    df["is_valid"] = ok.astype(int)

    # raw fitness = model-2 prediction (what you called C_FLP_sim_pred)
    df["fitness_raw"] = df[flp_col].astype(float)

    # scale valid genomes to [0, 1] (monotonic; preserves ordering)
    df["fitness"] = 0.0
    if ok.any():
        v = df.loc[ok, "fitness_raw"]
        vmin = float(v.min())
        vmax = float(v.max())
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            df.loc[ok, "fitness"] = (v - vmin) / (vmax - vmin)
        else:
            # all valid have identical score -> give them equal fitness
            df.loc[ok, "fitness"] = 1.0

    # helpful rank columns for inspection (valids first, then by fitness desc)
    df["_valid_rank"] = (~ok).astype(int)  # 0 for valid, 1 for invalid
    df = df.sort_values(["_valid_rank", "fitness"], ascending=[True, False])
    df["rank_overall"] = np.arange(1, len(df) + 1)
    df = df.drop(columns=["_valid_rank"])

    return df


# ============================================================
# Genome repair: enforce topology CNs, pt path, slots, expected edges
# ============================================================
def repair_genome(row: dict,
                  topo_specs: Dict[int, Dict],
                  embed_dir: str) -> dict:
    topo_id = int(row["topology_id"])
    spec = topo_specs[topo_id]

    # enforce topology CNs (authoritative)
    row["node1_connectivity"] = int(spec["node1_cn"])
    row["node2_connectivity"] = int(spec["node2_cn"])

    # If topology has only one node type, mirror node2 to node1 to avoid invalid CN↔ID pairs
    if int(spec.get("node_types", 2)) == 1:
        row["node2_connectivity"] = row["node1_connectivity"]
        # If node1_linker_id present, mirror it; else keep as-is (will be repaired later by population builder)
        if "node1_linker_id" in row:
            row["node2_linker_id"] = row.get("node1_linker_id", row.get("node2_linker_id"))

    # refresh topology_pt_path
    row["topology_pt_path"] = ensure_topology_pt(embed_dir, topo_id)

    # slots come from topology
    row["edge_slots"] = int(spec["edge_slots"])
    row["node_slots"] = int(spec["node_slots"])
    row["total_slots"] = int(spec["total_slots"])

    # expected edges from coverage + edge_slots
    cov = float(row["coverage_fraction"])
    if cov <= 0.0:
        cov = 0.25
    if cov not in ALLOWED_COVERAGE:
        # snap to nearest allowed value
        cov = min(ALLOWED_COVERAGE, key=lambda x: abs(x - cov))
    row["coverage_fraction"] = float(cov)

    n_fn, n_un = compute_expected_edges(int(row["edge_slots"]), cov)
    row["n_fn_edges_expected"] = int(n_fn)
    row["n_unfn_edges_expected"] = int(n_un)

    return row


def pick_node_linker(cn: int, pool3: List[int], pool4: List[int]) -> int:
    if int(cn) == 3:
        return int(random.choice(pool3))
    elif int(cn) == 4:
        return int(random.choice(pool4))
    else:
        # should not happen in your dataset
        raise ValueError(f"Unsupported node connectivity CN={cn}")


def pick_coverage() -> float:
    return float(random.choice(ALLOWED_COVERAGE))


# ============================================================
# Child creation (bundle-safe + CN-safe)
# ============================================================
def make_child(p1: dict,
               p2: dict,
               topo_specs: Dict[int, Dict],
               pool3: List[int], pool4: List[int],
               edge_bundles: pd.DataFrame,
               base_id: int,
               embed_dir: str,
               mutation_rate: float) -> dict:
    """
    Child creation with bundle-safe crossover and repair.
    Gene-groups:
      A) topology + node linkers
      B) edge bundle (parent_2c_id, edge_fn_id, bridge_type, base_id)
      C) coverage_fraction
    """

    child = {}

    # ---- A) topology + nodes (crossover)
    if random.random() < 0.5:
        child["topology_id"] = int(p1["topology_id"])
    else:
        child["topology_id"] = int(p2["topology_id"])

    # After topology fixed, assign node linkers consistent with CN
    spec = topo_specs[int(child["topology_id"])]
    node_types = int(spec.get("node_types", 2))
    cn1 = int(spec["node1_cn"])
    cn2 = int(spec["node2_cn"])

    # crossover: take node IDs from one parent
    if random.random() < 0.5:
        n1 = int(p1.get("node1_linker_id", -1))
        n2 = int(p1.get("node2_linker_id", -1))
        child["node1_linker_id"] = n1
        child["node2_linker_id"] = n2
    else:
        n1 = int(p2.get("node1_linker_id", -1))
        n2 = int(p2.get("node2_linker_id", -1))
        child["node1_linker_id"] = n1
        child["node2_linker_id"] = n2

    # Validate node linker IDs against required connectivities
    def _valid_node_id(cn_req: int, node_id: int) -> bool:
        pool = pool3 if cn_req == 3 else pool4
        return int(node_id) in set(pool)

    if not _valid_node_id(cn1, int(child["node1_linker_id"])):
        child["node1_linker_id"] = pick_node_linker(cn1, pool3, pool4)
    if not _valid_node_id(cn2, int(child["node2_linker_id"])):
        child["node2_linker_id"] = pick_node_linker(cn2, pool3, pool4)

    # If topology has only one node type, mirror node2 linker to node1 linker
    if node_types == 1:
        child["node2_linker_id"] = child["node1_linker_id"]

    # Repair node linkers if missing/invalid format; simplest is resample by CN
    if child["node1_linker_id"] is None or int(child["node1_linker_id"]) < 0:
        child["node1_linker_id"] = pick_node_linker(cn1, pool3, pool4)
    if child["node2_linker_id"] is None or int(child["node2_linker_id"]) < 0:
        child["node2_linker_id"] = pick_node_linker(cn2, pool3, pool4)

    # Mutation: node linker resample (CN-safe)
    if random.random() < mutation_rate:
        child["node1_linker_id"] = pick_node_linker(cn1, pool3, pool4)
    if random.random() < mutation_rate:
        child["node2_linker_id"] = pick_node_linker(cn2, pool3, pool4)
        if node_types == 1:
            child["node2_linker_id"] = child["node1_linker_id"]

    # ---- B) edge bundle (crossover, base-fixed)
    if random.random() < 0.5:
        child["parent_2c_id"] = int(p1["parent_2c_id"])
        child["edge_fn_id"] = int(p1["edge_fn_id"])
        child["bridge_type"] = str(p1["bridge_type"])
        child["base_id"] = int(base_id)
    else:
        child["parent_2c_id"] = int(p2["parent_2c_id"])
        child["edge_fn_id"] = int(p2["edge_fn_id"])
        child["bridge_type"] = str(p2["bridge_type"])
        child["base_id"] = int(base_id)

    # Repair: if the tuple doesn't exist in edge_bundles, resample a valid bundle
    # Build a quick lookup set
    # (edge bundles already filtered for base_id)
    # We check (edge_fn_id, parent_2c_id, bridge_type)
    if not hasattr(make_child, "_bundle_set_cache"):
        make_child._bundle_set_cache = {}

    cache_key = f"base_{int(base_id)}"
    if cache_key not in make_child._bundle_set_cache:
        s = set()
        for _, rr in edge_bundles.iterrows():
            s.add((int(rr["edge_fn_id"]), int(rr["parent_2c_id"]), str(rr["bridge_type"])))
        make_child._bundle_set_cache[cache_key] = s

    bset = make_child._bundle_set_cache[cache_key]
    tup = (int(child["edge_fn_id"]), int(child["parent_2c_id"]), str(child["bridge_type"]))
    if tup not in bset:
        bun = edge_bundles.sample(n=1).iloc[0]
        child["parent_2c_id"] = int(bun["parent_2c_id"])
        child["edge_fn_id"] = int(bun["edge_fn_id"])
        child["bridge_type"] = str(bun["bridge_type"])
        child["base_id"] = int(base_id)

    # Mutation: swap bundle (still base-fixed)
    if random.random() < mutation_rate:
        bun = edge_bundles.sample(n=1).iloc[0]
        child["parent_2c_id"] = int(bun["parent_2c_id"])
        child["edge_fn_id"] = int(bun["edge_fn_id"])
        child["bridge_type"] = str(bun["bridge_type"])
        child["base_id"] = int(base_id)

    # ---- C) coverage_fraction (crossover + mutation)
    if random.random() < 0.5:
        child["coverage_fraction"] = float(p1.get("coverage_fraction", 0.25))
    else:
        child["coverage_fraction"] = float(p2.get("coverage_fraction", 0.25))

    # snap coverage
    if child["coverage_fraction"] <= 0.0:
        child["coverage_fraction"] = 0.25
    if child["coverage_fraction"] not in ALLOWED_COVERAGE:
        child["coverage_fraction"] = float(min(ALLOWED_COVERAGE, key=lambda x: abs(x - child["coverage_fraction"])))

    # mutate coverage
    if random.random() < mutation_rate:
        child["coverage_fraction"] = pick_coverage()

    # topology-dependent fields (slots, pt paths, CNs, expected edges)
    child = repair_genome(child, topo_specs, embed_dir)

    return child


# ============================================================
# Predictor runner (calls your working predictor script)
# ============================================================
def run_predictor(predictor_py: str,
                  pop_csv: str,
                  out_csv: str,
                  device: str,
                  batch_size: int) -> None:
    cmd = [
        os.environ.get("PYTHON", "python"),
        predictor_py,
        "--pop_csv", pop_csv,
        "--out_csv", out_csv,
        "--device", device,
        "--batch_size", str(batch_size),
    ]
    print("Running predictor:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"Predictor failed (exit {r.returncode}). See output above.")


# ============================================================
# Main GA loop
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id", type=int, required=True)
    ap.add_argument("--gen0_csv", type=str, required=True)
    ap.add_argument("--topo", type=str, required=True)
    ap.add_argument("--edge_bundle", type=str, required=True)
    ap.add_argument("--feat_dir", type=str, required=True)
    ap.add_argument("--embed_dir", type=str, required=True)
    ap.add_argument("--predictor_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pld_min", type=float, default=12.0)
    ap.add_argument("--av_min", type=float, default=0.30)
    ap.add_argument("--selection_fraction", type=float, default=0.25)
    ap.add_argument("--mutation_rate", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    ensure_dir(args.out_dir)

    topo_specs = parse_topology_specs(args.topo)
    edge_bundles = load_edge_bundle_library(args.edge_bundle, args.base_id)

    # pools from feature files (MUST match what predictor/dataset uses)
    # node features:
    pool3 = load_id_pool(os.path.join(args.feat_dir, "3_con_linker_feat_ms.csv"), "linker_id")
    pool4 = load_id_pool(os.path.join(args.feat_dir, "4_con_linker_feat_ms.csv"), "linker_id")

    # Load gen0 population
    pop0 = pd.read_csv(args.gen0_csv)
    pop0 = pop0.to_dict(orient="records")

    # Ensure gen0 has all mandatory / repaired fields
    repaired0 = []
    for r in pop0:
        r["base_id"] = int(args.base_id)
        # enforce coverage allowed
        cov = float(r.get("coverage_fraction", 0.25))
        if cov <= 0.0:
            cov = 0.25
        if cov not in ALLOWED_COVERAGE:
            cov = float(min(ALLOWED_COVERAGE, key=lambda x: abs(x - cov)))
        r["coverage_fraction"] = cov
        r = repair_genome(r, topo_specs, args.embed_dir)
        repaired0.append(r)

    # generation loop
    current_pop = repaired0
    N = len(current_pop)
    if N == 0:
        raise RuntimeError("Gen0 population is empty.")

    for gen in range(args.generations):
        gen_dir = os.path.join(args.out_dir, f"gen_{gen:03d}")
        ensure_dir(gen_dir)

        pop_csv = os.path.join(gen_dir, "population.csv")
        pred_csv = os.path.join(gen_dir, "population_with_preds.csv")

        pd.DataFrame(current_pop).to_csv(pop_csv, index=False)

        # run predictor
        run_predictor(args.predictor_py, pop_csv, pred_csv, args.device, args.batch_size)

        dfp = pd.read_csv(pred_csv)

        # expected prediction cols used in your pipeline
        # model1: lcd_pred, pld_pred
        # model1b: av_pred (or L_AV_frac_pred)
        # model2: flp_pred
        # Adjust below names only if your predictor writes different ones.
        # (I am keeping your earlier convention)
        pld_col = "PLD_pred" if "PLD_pred" in dfp.columns else "pld_pred"
        av_col  = "L_AV_frac_pred" if "L_AV_frac_pred" in dfp.columns else ("av_pred" if "av_pred" in dfp.columns else "AV_pred")
        flp_col = "C_FLP_sim_pred" if "C_FLP_sim_pred" in dfp.columns else ("flp_pred" if "flp_pred" in dfp.columns else "C_FLP_sim_large_pred")

        if pld_col not in dfp.columns or av_col not in dfp.columns or flp_col not in dfp.columns:
            raise KeyError(
                f"Predictor output missing required cols.\n"
                f"Need PLD, AV, FLP predictions, tried: {pld_col}, {av_col}, {flp_col}\n"
                f"Columns available: {list(dfp.columns)}"
            )

        # fitness (ranking-based)
        dfp = compute_rank_fitness(dfp, pld_col, av_col, flp_col, args.pld_min, args.av_min)

        # save ranking summary
        dfp.sort_values("fitness", ascending=False).to_csv(os.path.join(gen_dir, "population_ranked.csv"), index=False)

        # selection
        k = max(2, int(math.ceil(args.selection_fraction * len(dfp))))
        elites = dfp.sort_values("fitness", ascending=False).head(k).to_dict(orient="records")

        # breed next generation (keep population size constant)
        next_pop: List[dict] = []

        # elitism: keep top few unchanged (repaired)
        n_elite_keep = min(len(elites), max(2, int(0.10 * N)))
        for i in range(n_elite_keep):
            rr = dict(elites[i])
            rr = repair_genome(rr, topo_specs, args.embed_dir)
            next_pop.append(rr)

        # breed remaining
        while len(next_pop) < N:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = make_child(
                p1, p2,
                topo_specs=topo_specs,
                pool3=pool3, pool4=pool4,
                edge_bundles=edge_bundles,
                base_id=args.base_id,
                embed_dir=args.embed_dir,
                mutation_rate=args.mutation_rate
            )
            next_pop.append(child)

        # final repair pass (belt-and-suspenders)
        repaired_next = []
        for rr in next_pop:
            rr["base_id"] = int(args.base_id)
            rr = repair_genome(rr, topo_specs, args.embed_dir)
            repaired_next.append(rr)

        current_pop = repaired_next

        # write gen+1 population immediately (so crash later still keeps output)
        if gen + 1 < args.generations:
            gen_next_dir = os.path.join(args.out_dir, f"gen_{gen+1:03d}")
            ensure_dir(gen_next_dir)
            pd.DataFrame(current_pop).to_csv(os.path.join(gen_next_dir, "population.csv"), index=False)

    print(f"\n✅ GA finished: {args.out_dir}")


if __name__ == "__main__":
    main()

