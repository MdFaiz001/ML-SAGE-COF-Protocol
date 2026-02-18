#!/usr/bin/env python3
"""
COF → lammps-interface pipeline with:

- Parallel execution (full CPU utilization)
- Sorting by n_atoms (smallest first)
- RESUME SUPPORT (Option A):
    * Skip COFs already in success log, OR
    * Skip COFs whose data.COF_xxxx and in.COF_xxxx exist in filtered_cofs_ready
- LIVE LOGGING:
    * Append to log/error/success CSVs as each job finishes
- LIVE COPYING:
    * Immediately copy successful COFs into cofs_for_optimization/
    * Immediately write template-based in.COF

Inputs:
    filtered_cofs_ready/ (folders: topo__COF_XXXXXX)
    final_cofs_after_rmsd_and_cell_filter.csv

Usage:
    python prepare_lammps_inputs_resume_live.py
"""

import os
import shutil
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


# ===================== USER SETTINGS ===================== #

COF_ROOT    = "filtered_cofs_ready"
FINAL_CSV   = "final_cofs_after_rmsd_and_cell_filter.csv"

FORCE_FIELD = "Dreiding"     # or "UFF", etc.
NPROCS      = 44             # number of parallel workers

TEMPLATE_INCOF = "./in.COF"  # global LAMMPS template
OPT_ROOT       = "cofs_for_optimization"

LOG_CSV     = "lammps_interface_log.csv"
ERR_CSV     = "lammps_interface_errors.csv"
SUCCESS_CSV = "lammps_interface_success.csv"


# ===================== TEMPLATE LOADING ===================== #

def load_template_lines():
    if not os.path.isfile(TEMPLATE_INCOF):
        raise FileNotFoundError(f"Template in.COF not found: {TEMPLATE_INCOF}")
    with open(TEMPLATE_INCOF, "r") as f:
        return f.readlines()

TEMPLATE_LINES = load_template_lines()


# ===================== SMALL HELPERS ===================== #

def find_cif(folder_path: str):
    """Return the full path to the first .cif in folder_path, or None."""
    if not os.path.isdir(folder_path):
        return None
    for f in os.listdir(folder_path):
        if f.lower().endswith(".cif"):
            return os.path.join(folder_path, f)
    return None


def append_row_csv(path: str, row: dict):
    """
    Append a single row (dict) to a CSV file.
    Create the file with header if it doesn't exist.
    """
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ===================== LAMMPS-INTERFACE WORKER ===================== #

def run_job(job):
    """
    Run lammps-interface for a single COF and return a result dict.

    job = {
        "cof_id": ...,
        "folder": ...,
        "n_atoms": ...
    }
    """
    cof_id  = job["cof_id"]
    folder  = job["folder"]
    n_atoms = job.get("n_atoms", None)

    folder_path = os.path.join(COF_ROOT, folder)

    if not os.path.isdir(folder_path):
        return {
            "cof_id": cof_id,
            "folder": folder,
            "n_atoms": n_atoms,
            "status": "NO_FOLDER",
            "error": f"Folder not found: {folder_path}",
            "has_data": False,
            "has_incof": False,
            "time_sec": 0.0,
        }

    cif_path = find_cif(folder_path)
    if cif_path is None:
        return {
            "cof_id": cof_id,
            "folder": folder,
            "n_atoms": n_atoms,
            "status": "NO_CIF",
            "error": "No CIF file in folder",
            "has_data": False,
            "has_incof": False,
            "time_sec": 0.0,
        }

    cif_name = os.path.basename(cif_path)

    cmd = [
        "lammps-interface",
        "--force_field", FORCE_FIELD,
        "--minimize",
        cif_name,
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=folder_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - start
    except Exception as e:
        return {
            "cof_id": cof_id,
            "folder": folder,
            "n_atoms": n_atoms,
            "status": "EXCEPTION",
            "error": str(e),
            "has_data": False,
            "has_incof": False,
            "time_sec": 0.0,
        }

    # Determine expected output filenames
    cof_num    = cof_id.split("_")[-1]
    data_name  = f"data.COF_{cof_num}"
    incof_name = f"in.COF_{cof_num}"

    data_path  = os.path.join(folder_path, data_name)
    incof_path = os.path.join(folder_path, incof_name)

    has_data  = os.path.isfile(data_path)
    has_incof = os.path.isfile(incof_path)

    if proc.returncode == 0 and has_data and has_incof:
        status = "OK"
        error  = ""
    else:
        status = "FAILED"
        error  = proc.stderr.strip()

    return {
        "cof_id": cof_id,
        "folder": folder,
        "n_atoms": n_atoms,
        "status": status,
        "error": error,
        "has_data": has_data,
        "has_incof": has_incof,
        "time_sec": elapsed,
    }


# ===================== WRITE NEW in.COF ===================== #

def write_new_incof(dest_folder: str, cof_id: str):
    """
    In dest_folder:
      - remove in.COF_XXXX (if present)
      - write a new in.COF based on TEMPLATE_LINES
        with correct read_data/write_data names.
    """
    cof_num     = cof_id.split("_")[-1]
    data_name   = f"data.COF_{cof_num}"
    relaxed_out = f"relaxed_COF_{cof_num}.data"

    old_incof = os.path.join(dest_folder, f"in.COF_{cof_num}")
    if os.path.exists(old_incof):
        os.remove(old_incof)

    new_incof = os.path.join(dest_folder, "in.COF")

    out_lines = []
    for line in TEMPLATE_LINES:
        if "read_data" in line:
            out_lines.append(f"read_data       {data_name}\n")
        elif "write_data" in line:
            out_lines.append(f"write_data      {relaxed_out}\n")
        else:
            out_lines.append(line)

    with open(new_incof, "w") as f:
        f.writelines(out_lines)


def copy_and_prepare_opt_folder(cof_id: str, folder: str):
    """
    Copy the COF folder from COF_ROOT to OPT_ROOT and prepare final in.COF.
    Always OVERWRITES existing folder, as per Option A.
    """
    src_dir = os.path.join(COF_ROOT, folder)
    dst_dir = os.path.join(OPT_ROOT, folder)

    if not os.path.isdir(src_dir):
        print(f"[WARN] Source folder missing for {cof_id}: {src_dir}")
        return

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)
    write_new_incof(dst_dir, cof_id)


# ===================== MAIN ===================== #

def main():

    # ---------- Load final COF list ----------
    if not os.path.isfile(FINAL_CSV):
        raise FileNotFoundError(f"Missing final COF list CSV: {FINAL_CSV}")

    df_all = pd.read_csv(FINAL_CSV)
    df_all["cof_id"]     = df_all["cof_id"].astype(str)
    df_all["output_dir"] = df_all["output_dir"].astype(str)

    if "n_atoms" in df_all.columns:
        df_all = df_all.sort_values("n_atoms", ascending=True)

    total_all = len(df_all)

    # ---------- Existing success log ----------
    if os.path.isfile(SUCCESS_CSV):
        df_succ_prev = pd.read_csv(SUCCESS_CSV)
        df_succ_prev["cof_id"] = df_succ_prev["cof_id"].astype(str)
        success_ids_prev = set(df_succ_prev["cof_id"].tolist())
    else:
        df_succ_prev = pd.DataFrame()
        success_ids_prev = set()

    # ---------- File-based successes (data+in in filtered_cofs_ready) ----------
    file_success_ids = set()
    for _, row in df_all.iterrows():
        cof_id = row["cof_id"]
        folder = row["output_dir"]
        folder_path = os.path.join(COF_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue

        cof_num    = cof_id.split("_")[-1]
        data_name  = f"data.COF_{cof_num}"
        incof_name = f"in.COF_{cof_num}"

        data_path  = os.path.join(folder_path, data_name)
        incof_path = os.path.join(folder_path, incof_name)

        if os.path.isfile(data_path) and os.path.isfile(incof_path):
            file_success_ids.add(cof_id)

    success_ids_initial = success_ids_prev.union(file_success_ids)

    # ---------- Build job list: skip already-successful ----------
    jobs = []
    for _, row in df_all.iterrows():
        cof_id = row["cof_id"]
        if cof_id in success_ids_initial:
            continue
        jobs.append({
            "cof_id": cof_id,
            "folder": row["output_dir"],
            "n_atoms": row.get("n_atoms", None),
        })

    total_remaining = len(jobs)

    print(f"[INFO] Total COFs in CSV            : {total_all}")
    print(f"[INFO] Already successful (log/files): {len(success_ids_initial)}")
    print(f"[INFO] To run in this session       : {total_remaining}")
    print(f"[INFO] Using {NPROCS} parallel workers.\n")

    # ---------- Ensure OPT_ROOT exists ----------
    if not os.path.exists(OPT_ROOT):
        os.makedirs(OPT_ROOT)

    # ---------- Also copy initial-success COFs immediately ----------
    # (So OPT_ROOT is always consistent with known successes)
    id_to_folder = dict(zip(df_all["cof_id"], df_all["output_dir"]))
    for cof_id in sorted(success_ids_initial):
        folder = id_to_folder.get(cof_id, None)
        if folder is not None:
            copy_and_prepare_opt_folder(cof_id, folder)

    # ---------- Process jobs with live logging & copying ----------
    new_success_ids = set()

    if total_remaining > 0:
        with ProcessPoolExecutor(max_workers=NPROCS) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in jobs}

            for future in tqdm(as_completed(future_to_job),
                               total=total_remaining,
                               desc="Running lammps-interface"):
                res = future.result()

                # Always append to master log
                append_row_csv(LOG_CSV, res)

                if res["status"] == "OK":
                    # Live success log
                    append_row_csv(SUCCESS_CSV, res)
                    new_success_ids.add(res["cof_id"])

                    # Live copying to OPT_ROOT
                    copy_and_prepare_opt_folder(res["cof_id"], res["folder"])
                else:
                    # Log errors / non-OK
                    append_row_csv(ERR_CSV, res)
    else:
        print("[INFO] No new COFs to run; all already successful.\n")

    # ---------- Summary ----------
    overall_success_ids = success_ids_initial.union(new_success_ids)

    print("\n========== LAMMPS-INTERFACE SUMMARY ==========")
    print(f"Total COFs in CSV          : {total_all}")
    print(f"Previously successful      : {len(success_ids_initial)}")
    print(f"Newly successful this run  : {len(new_success_ids)}")
    print(f"Total successful overall   : {len(overall_success_ids)}")
    print("Log files:")
    print(f"  All   : {LOG_CSV}")
    print(f"  Error : {ERR_CSV}")
    print(f"  OK    : {SUCCESS_CSV}")
    print("==============================================\n")

    print("[INFO] All successful COFs now exist in:", OPT_ROOT)
    print("[INFO] You can safely rerun this script anytime; it will resume and skip completed COFs.\n")


if __name__ == "__main__":
    main()

