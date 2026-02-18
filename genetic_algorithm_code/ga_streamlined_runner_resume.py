#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime


# -------------------------
# Utilities
# -------------------------

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)

def run_cmd(cmd, cwd: Path, log_file: Path):
    """Run a command and tee stdout/stderr into a log file."""
    log(f"RUN: {' '.join(cmd)} (cwd={cwd})")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {ts()} RUN: {' '.join(cmd)} (cwd={cwd}) =====\n")
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        ret = p.wait()
        f.write(f"===== {ts()} EXIT CODE: {ret} =====\n")
    if ret != 0:
        raise RuntimeError(f"Command failed (cwd={cwd}): {' '.join(cmd)}. See log: {log_file}")

def must_exist(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"[MISSING] {what}: {path}")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_into(src: Path, dst_dir: Path):
    must_exist(src, f"Required file")
    safe_mkdir(dst_dir)
    shutil.copy2(src, dst_dir / src.name)

def marker(path: Path):
    path.write_text(ts() + "\n", encoding="utf-8")

def has_marker(path: Path) -> bool:
    return path.exists()

def count_cifs(out_root: Path) -> int:
    """Count CIF files in generated_cofs_new1/<output_dir>/*.cif"""
    if not out_root.exists():
        return 0
    n = 0
    for sub in out_root.iterdir():
        if sub.is_dir():
            n += len(list(sub.glob("*.cif")))
    return n

def summarize_generated(gen_dir: Path):
    out_root = gen_dir / "generated_cofs_new1"
    plan_csv = gen_dir / "COF_generation_plan.csv"
    plan_n = len(pd.read_csv(plan_csv)) if plan_csv.exists() else 0
    cif_n = count_cifs(out_root)
    log(f"[SUMMARY] Plan COFs: {plan_n} | CIFs produced: {cif_n} | out_root: {out_root}")

def summarize_filter(gen_dir: Path):
    fcsv = gen_dir / "final_cofs_after_rmsd_and_cell_filter.csv"
    mcsv = gen_dir / "cof_master_status.csv"
    if fcsv.exists():
        n = len(pd.read_csv(fcsv))
        log(f"[SUMMARY] Filter passed COFs: {n} (final_cofs_after_rmsd_and_cell_filter.csv)")
    else:
        log("[SUMMARY] Filter output not found yet.")
    if mcsv.exists():
        log("[SUMMARY] cof_master_status.csv present.")
    else:
        log("[SUMMARY] cof_master_status.csv missing.")

def summarize_lammps(gen_dir: Path):
    succ = gen_dir / "lammps_interface_success.csv"
    err  = gen_dir / "lammps_interface_errors.csv"
    if succ.exists():
        ns = len(pd.read_csv(succ))
    else:
        ns = 0
    if err.exists():
        ne = len(pd.read_csv(err))
    else:
        ne = 0
    log(f"[SUMMARY] LAMMPS-interface success: {ns} | errors: {ne}")

def ensure_link_or_copy_dir(src: Path, dst: Path):
    must_exist(src, f"Required directory")
    if dst.exists():
        return
    try:
        os.symlink(str(src.resolve()), str(dst))
        log(f"Symlinked dir: {dst.name} -> {src}")
    except Exception:
        shutil.copytree(src, dst)
        log(f"Copied dir: {dst.name} -> {src}")

def ensure_copy_file(src: Path, dst: Path):
    must_exist(src, f"Required file")
    if dst.exists():
        return
    shutil.copy2(src, dst)


# -------------------------
# Plan builder
# -------------------------

def make_generation_plan(pop_csv: Path,
                         out_plan_csv: Path,
                         topo_final_csv: Path,
                         l2c_csv: Path,
                         l3c_csv: Path,
                         l4c_csv: Path,
                         edge_bundle_csv: Path):
    pop = pd.read_csv(pop_csv)

    topo = pd.read_csv(topo_final_csv).set_index("topology_id")
    l2c  = pd.read_csv(l2c_csv).set_index("linker_id")
    l3c  = pd.read_csv(l3c_csv).set_index("linker_id")
    l4c  = pd.read_csv(l4c_csv).set_index("linker_id")
    edge = pd.read_csv(edge_bundle_csv).set_index("edge_fn_id")

    def map_node_name(connectivity: int, linker_id: int) -> int:
        connectivity = int(connectivity)
        linker_id = int(linker_id)
        if connectivity == 3:
            return int(l3c.loc[linker_id, "name"])
        if connectivity == 4:
            return int(l4c.loc[linker_id, "name"])
        raise ValueError(f"Unsupported node connectivity: {connectivity}")

    rows = []
    for _, r in pop.iterrows():
        tid = int(r["topology_id"])
        topo_name = str(topo.loc[tid, "Topology"])
        ntypes = int(topo.loc[tid, "# Node types"])
        cof_id = str(r["cof_id"])
        output_dir = f"{topo_name}__{cof_id}"

        out = {
            "cof_id": cof_id,
            "output_dir": output_dir,
            "topology_name": topo_name,
            "num_node_types": ntypes,
            "node1_type": f"{int(r['node1_connectivity'])}C",
            "node1_linker": map_node_name(r["node1_connectivity"], r["node1_linker_id"]),
            "coverage_pct": float(r["coverage_fraction"]) * 100.0,
            "parent_2c": str(l2c.loc[int(r["parent_2c_id"]), "name"]),
            "edge_fn_name": str(edge.loc[int(r["edge_fn_id"]), "fn_name"]),
        }

        if ntypes == 2:
            out.update({
                "node2_type": f"{int(r['node2_connectivity'])}C",
                "node2_linker": map_node_name(r["node2_connectivity"], r["node2_linker_id"]),
            })

        rows.append(out)

    plan = pd.DataFrame(rows)
    plan.to_csv(out_plan_csv, index=False)
    return plan


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--population_csv", required=True)
    ap.add_argument("--scripts_dir", default=".")
    ap.add_argument("--embed_dir", default="embed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force_stage", choices=["plan", "pormake", "filter", "lammps", "predict", "all"],
                    default=None)

    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    sdir = Path(args.scripts_dir)
    safe_mkdir(gen_dir)

    main_log = gen_dir / "ga_pipeline.log"

    m_plan   = gen_dir / ".done_plan"
    m_build  = gen_dir / ".done_pormake"
    m_filter = gen_dir / ".done_filter"
    m_lmp    = gen_dir / ".done_lammps"
    m_pred   = gen_dir / ".done_predict"

    def should_redo(stage_name: str) -> bool:
        return args.force_stage in ("all", stage_name)

    log("Preparing generation directory (copying scripts/deps)...")

    for f in ["new_generate_cofs_pormake_timed.py", "all_in_one_filter_n1.py",
              "prepare_lammps_inputs_resume_live.py", "predict_all_models_ga_population.py"]:
        src = sdir / f
        if src.exists():
            shutil.copy2(src, gen_dir / f)

    predictor_modules = [
        "dataset.py", "model.py",
        "dataset_model1b.py", "model_model1b.py",
        "dataset_model2_v2.py", "model_model2_v2.py",
    ]
    for f in predictor_modules:
        src = sdir / f
        must_exist(src, f"Predictor module required by imports")
        shutil.copy2(src, gen_dir / f)

    for f in ["topo_sorted.csv", "topo_final.csv",
              "2c_linkers.csv", "3c_linkers.csv", "4c_linkers.csv",
              "functionalized_2c_linkers.csv", "edge_bundle_library.csv"]:
        copy_into(sdir / f, gen_dir)

    for d in ["2c_xyz", "3c_xyz", "4c_xyz", "fn2c_xyz"]:
        src = sdir / d
        dst = gen_dir / d
        must_exist(src, f"XYZ directory {d} (needed by PORMAKE)")
        if not dst.exists():
            try:
                os.symlink(str(src.resolve()), str(dst))
                log(f"Symlinked {d} -> {src}")
            except Exception:
                shutil.copytree(src, dst)
                log(f"Copied {d} -> {src}")

    if (gen_dir / "in.COF").exists():
        log("Found in.COF in gen_dir.")
    else:
        must_exist(sdir / "in.COF", "LAMMPS template in.COF (put it in scripts_dir)")
        shutil.copy2(sdir / "in.COF", gen_dir / "in.COF")
        log("Copied in.COF into gen_dir.")

    ensure_link_or_copy_dir(sdir / "features", gen_dir / "features")
    ensure_link_or_copy_dir(sdir / args.embed_dir, gen_dir / args.embed_dir)
    for ckpt in ["model1_final.pt", "model1b_L_avfrac_final.pt", "model2_v2_flp_large_final.pt"]:
        ensure_copy_file(sdir / ckpt, gen_dir / ckpt)

    # Stage PLAN
    plan_csv = gen_dir / "COF_generation_plan.csv"
    if should_redo("plan") or not has_marker(m_plan) or not plan_csv.exists():
        log("Stage PLAN: building COF_generation_plan.csv ...")
        make_generation_plan(
            pop_csv=Path(args.population_csv),
            out_plan_csv=plan_csv,
            topo_final_csv=gen_dir / "topo_final.csv",
            l2c_csv=gen_dir / "2c_linkers.csv",
            l3c_csv=gen_dir / "3c_linkers.csv",
            l4c_csv=gen_dir / "4c_linkers.csv",
            edge_bundle_csv=gen_dir / "edge_bundle_library.csv",
        )
        marker(m_plan)
        log("Stage PLAN: done.")
    else:
        log("Stage PLAN: already done (marker present).")

    # Stage PORMAKE
    out_root = gen_dir / "generated_cofs_new1"
    rmsd_src = out_root / "rmsd_log.csv"
    cov_src  = out_root / "coverage_log.csv"

    if should_redo("pormake") or not has_marker(m_build):
        log("Stage PORMAKE: running new_generate_cofs_pormake_timed.py ...")
        run_cmd([sys.executable, "new_generate_cofs_pormake_timed.py"], cwd=gen_dir, log_file=main_log)
        must_exist(rmsd_src, "PORMAKE output rmsd_log.csv in generated_cofs_new1/")
        shutil.copy2(rmsd_src, gen_dir / "rmsd_log.csv")
        if cov_src.exists():
            shutil.copy2(cov_src, gen_dir / "coverage_log.csv")
        marker(m_build)
        summarize_generated(gen_dir)
        log("Stage PORMAKE: done.")
    else:
        log("Stage PORMAKE: already done (marker present).")
        summarize_generated(gen_dir)

    # Stage FILTER
    filt_out = gen_dir / "final_cofs_after_rmsd_and_cell_filter.csv"
    if should_redo("filter") or not has_marker(m_filter) or not filt_out.exists():
        log("Stage FILTER: running all_in_one_filter_n1.py ...")
        must_exist(gen_dir / "rmsd_log.csv", "rmsd_log.csv in gen_dir")
        run_cmd([sys.executable, "all_in_one_filter_n1.py"], cwd=gen_dir, log_file=main_log)
        marker(m_filter)
        summarize_filter(gen_dir)
        log("Stage FILTER: done.")
    else:
        log("Stage FILTER: already done (marker present).")
        summarize_filter(gen_dir)

    # Stage LAMMPS
    lsucc = gen_dir / "lammps_interface_success.csv"
    if should_redo("lammps") or not has_marker(m_lmp) or not lsucc.exists():
        log("Stage LAMMPS: running prepare_lammps_inputs_resume_live.py ...")
        must_exist(gen_dir / "in.COF", "in.COF template in gen_dir")
        must_exist(gen_dir / "final_cofs_after_rmsd_and_cell_filter.csv", "filter output csv")
        run_cmd([sys.executable, "prepare_lammps_inputs_resume_live.py"], cwd=gen_dir, log_file=main_log)
        marker(m_lmp)
        summarize_lammps(gen_dir)
        log("Stage LAMMPS: done.")
    else:
        log("Stage LAMMPS: already done (marker present).")
        summarize_lammps(gen_dir)

    # Stage PREDICT  ✅ FIXED PATHS HERE
    pred_out = gen_dir / "population_with_preds.csv"
    if should_redo("predict") or not has_marker(m_pred) or not pred_out.exists():
        log("Stage PREDICT: preparing population_for_ml.csv ...")
        if lsucc.exists():
            succ = pd.read_csv(lsucc)
            ok_ids = set(succ["cof_id"].astype(str).tolist()) if "cof_id" in succ.columns else set()
        else:
            ok_ids = set()

        pop = pd.read_csv(args.population_csv)
        pop_ok = pop[pop["cof_id"].astype(str).isin(ok_ids)].copy()
        pop_ok.to_csv(gen_dir / "population_for_ml.csv", index=False)

        if len(pop_ok) == 0:
            log("Stage PREDICT: no survivors; writing empty population_with_preds.csv")
            pred_out.write_text("", encoding="utf-8")
            marker(m_pred)
        else:
            log(f"Stage PREDICT: running predictor on {len(pop_ok)} COFs ...")
            must_exist(gen_dir / "population_for_ml.csv", "population_for_ml.csv in gen_dir")

            # IMPORTANT: because cwd=gen_dir, pass relative filenames (NOT gen_dir/...)
            run_cmd([
                sys.executable, "predict_all_models_ga_population.py",
                "--pop_csv", "population_for_ml.csv",
                "--embed_dir", args.embed_dir,
                "--out_csv", "population_with_preds.csv",
                "--device", args.device,
                "--batch_size", str(args.batch_size),
            ], cwd=gen_dir, log_file=main_log)

            marker(m_pred)

        log("Stage PREDICT: done.")
    else:
        log("Stage PREDICT: already done (marker present).")

    log("✅ ALL DONE. You can safely rerun this script anytime; it will resume.")


if __name__ == "__main__":
    main()

