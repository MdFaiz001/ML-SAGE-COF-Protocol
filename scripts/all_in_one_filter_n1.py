#!/usr/bin/env python3
"""
FULL COF FILTERING + MASTER STATUS PIPELINE (ONE SCRIPT)

Performs the following steps automatically:

1) RMSD FEASIBILITY FILTER
   - Reads rmsd_log.csv
   - A COF is RMSD-feasible if:
        * 1-node topology → node0 feasible = TRUE
        * >1-node topology → ALL node_type_idx for that cof_id are feasible = TRUE

2) CHECK CIF EXISTENCE
   - Ensures COF folders inside generated_cofs_new1/<output_dir>
     contain <cof_id>.cif
   - Missing CIF is interpreted as a TIMEOUT in PORMake

3) EXTRACT CELL PARAMETERS DIRECTLY FROM CIF
   - a, b, c
   - alpha, beta, gamma
   - number of atoms

4) APPLY CELL GEOMETRY FILTER:
   * a, b, c >= 10 Å
   * Anisotropy (max/min) <= 10
   * All angles in [60°, 140°]
   * Allowed angle patterns:
       (A) Orthorhombic-like → all ~90°
       (B) Monoclinic-like   → exactly two ~90° + third ∈ [60°,130°]
       (C) Hexagonal-like    → α≈90°, β≈90°, γ≈120°

5) SORT FINAL COFs BY n_atoms ASCENDING

6) COPY FILTERED COF DIRECTORIES TO:
       filtered_cofs_ready/<COF folders>

7) MASTER STATUS CSV
   - Writes cof_master_status.csv with one row per COF in COF_generation_plan.csv
   - Columns include:
       cof_id, output_dir,
       rmsd_status, cif_status, cell_status,
       final_status, failure_reason,
       a, b, c, alpha, beta, gamma, n_atoms, cif_path

Final Outputs:
    - final_cofs_after_rmsd_and_cell_filter.csv  (only COFs that pass EVERYTHING)
    - cof_master_status.csv                      (ALL planned COFs with status)
    - filtered_cofs_ready/<COF folders>
"""

import os
import shutil
import pandas as pd
import numpy as np

# ======================================================================
# USER SETTINGS
# ======================================================================

OUTPUT_ROOT = "./generated_cofs_new1"
PLAN_CSV    = "COF_generation_plan.csv"
RMSD_CSV    = "rmsd_log.csv"

FINAL_OUT   = "final_cofs_after_rmsd_and_cell_filter.csv"
MASTER_OUT  = "cof_master_status.csv"
COPY_ROOT   = "./filtered_cofs_ready"

TOL90  = 5.0
TOL120 = 5.0


# ======================================================================
# 1. CIF PARAMETER EXTRACTION
# ======================================================================

def extract_cell_parameters(cif_path):
    """Extract a,b,c,alpha,beta,gamma,n_atoms from a CIF file."""
    a = b = c = alpha = beta = gamma = None
    n_atoms = 0
    in_atom_block = False

    with open(cif_path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()

            # Lattice lengths
            if s.startswith("_cell_length_a"):
                a = float(s.split()[1])
            elif s.startswith("_cell_length_b"):
                b = float(s.split()[1])
            elif s.startswith("_cell_length_c"):
                c = float(s.split()[1])

            # Lattice angles
            elif s.startswith("_cell_angle_alpha"):
                alpha = float(s.split()[1])
            elif s.startswith("_cell_angle_beta"):
                beta = float(s.split()[1])
            elif s.startswith("_cell_angle_gamma"):
                gamma = float(s.split()[1])

            # Atom block starts
            if "_atom_type_partial_charge" in s:
                in_atom_block = True
                continue

            # Atom block ends
            if in_atom_block and s.startswith("loop_"):
                break

            # Count atoms
            if in_atom_block and s:
                parts = s.split()
                if len(parts) >= 2:
                    n_atoms += 1

    return a, b, c, alpha, beta, gamma, n_atoms


# ======================================================================
# 2. RMSD FEASIBILITY
# ======================================================================

def compute_rmsd_status():
    """
    Read RMSD_CSV and compute:
      - feasible_ids: COFs that pass RMSD (using your 1-node / 2-node logic)
      - rmsd_ids_all: all COF IDs that appear in rmsd_log.csv
    """
    df = pd.read_csv(RMSD_CSV)
    # Normalise feasibility flag
    df["feasible"] = df["feasible"].astype(str).str.upper().eq("TRUE")

    feasible_ids = set()
    rmsd_ids_all = set()

    for cof_id, grp in df.groupby("cof_id"):
        cof_id_str = str(cof_id)
        rmsd_ids_all.add(cof_id_str)

        # 1-node topology
        if grp["node_type_idx"].nunique() == 1:
            if grp["feasible"].iloc[0]:
                feasible_ids.add(cof_id_str)

        # >1-node topology: ALL node types must be feasible
        else:
            if grp["feasible"].all():
                feasible_ids.add(cof_id_str)

    return feasible_ids, rmsd_ids_all


# ======================================================================
# 3. CIF EXISTENCE
# ======================================================================

def existing_cif_ids(df_plan):
    existing_ids = set()
    cif_paths    = {}

    for _, row in df_plan.iterrows():
        cof_id = str(row["cof_id"])
        outdir = row["output_dir"]

        cif_path = os.path.join(OUTPUT_ROOT, str(outdir), f"{cof_id}.cif")

        if os.path.isfile(cif_path):
            existing_ids.add(cof_id)
            cif_paths[cof_id] = cif_path

    return existing_ids, cif_paths


# ======================================================================
# 4. CELL FILTER HELPERS
# ======================================================================

def near_90(x):  return abs(x - 90)  <= TOL90
def near_120(x): return abs(x - 120) <= TOL120

def cell_angle_pattern(alpha, beta, gamma):
    """Return True if passes orthorhombic / monoclinic / hexagonal patterns."""

    # --- Orthorhombic-like ---
    if near_90(alpha) and near_90(beta) and near_90(gamma):
        return True

    # --- Monoclinic-like ---
    angles = np.array([alpha, beta, gamma])
    flags  = [near_90(a) for a in angles]

    if sum(flags) == 2:
        # the remaining angle must be between 60 and 130
        for i, fl in enumerate(flags):
            if not fl:  # the non-90° angle
                if 60 <= angles[i] <= 130:
                    return True

    # --- Hexagonal-like ---
    if near_90(alpha) and near_90(beta) and near_120(gamma):
        return True

    return False


def cell_filter(a, b, c, alpha, beta, gamma):
    """Return True if cell parameters satisfy our filtering rules."""

    # Length sanity
    if min(a, b, c) < 10:
        return False

    # Anisotropy sanity (max/min <= 10)
    if max(a, b, c) / min(a, b, c) > 10:
        return False

    # Angle sanity
    for ang in (alpha, beta, gamma):
        if not (60 <= ang <= 140):
            return False

    # Pattern match
    return cell_angle_pattern(alpha, beta, gamma)


# ======================================================================
# 5. MAIN PIPELINE
# ======================================================================

def main():

    # ---------------------- Load plan -------------------------
    df_plan = pd.read_csv(PLAN_CSV)
    df_plan["cof_id"] = df_plan["cof_id"].astype(str)

    ids_in_plan = set(df_plan["cof_id"].tolist())
    print(f"[INFO] Total COFs in plan: {len(ids_in_plan)}")

    # ---------------------- RMSD filter -----------------------
    feasible_ids, rmsd_ids_all = compute_rmsd_status()
    print(f"[INFO] COFs with RMSD entries: {len(rmsd_ids_all)}")
    print(f"[INFO] RMSD-feasible COFs: {len(feasible_ids)}")

    # ---------------------- CIF existence ---------------------
    exist_ids, cif_paths = existing_cif_ids(df_plan)
    print(f"[INFO] COFs with existing CIFs: {len(exist_ids)}")

    # Intersection → only COFs that pass both RMSD & CIF existence
    initial_ids = feasible_ids.intersection(exist_ids)
    print(f"[INFO] After RMSD + CIF existence filter: {len(initial_ids)}")

    # ---------------------- Cell filter + final list ----------

    cell_info = {}   # cof_id -> dict with cell_status, reason, params, n_atoms, cif_path
    final_rows = []  # for final_cofs_after_rmsd_and_cell_filter.csv

    # We only apply cell filter to COFs that are RMSD-feasible AND have a CIF
    df_initial = df_plan[df_plan["cof_id"].isin(initial_ids)]

    for _, row in df_initial.iterrows():
        cof_id = row["cof_id"]
        cif_path = cif_paths[cof_id]

        a, b, c, alpha, beta, gamma, n_atoms = extract_cell_parameters(cif_path)

        # Default cell info dictionary for this COF
        info = {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_atoms": n_atoms,
            "cif_path": cif_path,
            "cell_status": None,
            "cell_reason": None,
        }

        # If any cell param is missing, mark as cell failure
        if None in (a, b, c, alpha, beta, gamma):
            info["cell_status"] = "fail"
            info["cell_reason"] = "missing_cell_params"
            cell_info[cof_id] = info
            continue

        # Apply cell filter
        if cell_filter(a, b, c, alpha, beta, gamma):
            info["cell_status"] = "ok"
            info["cell_reason"] = "ok"
            cell_info[cof_id] = info

            # This COF is part of final filtered list
            d = row.to_dict()
            d.update({
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "n_atoms": n_atoms,
                "cif_path": cif_path
            })
            final_rows.append(d)
        else:
            info["cell_status"] = "fail"
            info["cell_reason"] = "bad_cell"
            cell_info[cof_id] = info

    # ---------------------- Build final CSV (unchanged behaviour) -----
    df_out = pd.DataFrame(final_rows)

    if not df_out.empty:
        df_out = df_out.sort_values("n_atoms", ascending=True)

    # Save final filtered CSV
    df_out.to_csv(FINAL_OUT, index=False)
    print("\n========== FINAL SUMMARY ==========")
    print(f"Final COFs after ALL filters: {len(df_out)}")
    print(f"Saved: {FINAL_OUT}")
    print("===================================\n")

    # ---------------------- COPY DIRECTORIES ---------------------------
    if not os.path.exists(COPY_ROOT):
        os.makedirs(COPY_ROOT)

    print(f"[INFO] Copying selected COFs to: {COPY_ROOT}")

    for _, row in df_out.iterrows():
        outdir = row["output_dir"]
        src = os.path.join(OUTPUT_ROOT, str(outdir))
        dst = os.path.join(COPY_ROOT, str(outdir))

        if os.path.exists(dst):
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

    print("[INFO] Directory copy complete.")
    print("[INFO] COFs ready for LAMMPS optimization.\n")

    # ==================================================================
    # 6. MASTER STATUS CSV (ALL COFs)
    # ==================================================================
    print("[INFO] Building master status table for ALL COFs...")

    master_rows = []

    for _, row in df_plan.iterrows():
        cof_id = row["cof_id"]
        outdir = row["output_dir"]

        # ---------- RMSD status ----------
        if cof_id in feasible_ids:
            rmsd_status = "ok"
        elif cof_id in rmsd_ids_all:
            rmsd_status = "fail"
        else:
            rmsd_status = "not_found"  # no RMSD entry

        # ---------- CIF status ----------
        if cof_id in exist_ids:
            cif_status = "ok"
        else:
            cif_status = "missing"  # interpret as timeout / not generated

        # ---------- Cell status & parameters ----------
        info = cell_info.get(cof_id, None)

        if info is not None:
            cell_status = info["cell_status"]
            cell_reason = info["cell_reason"]
            a = info["a"]
            b = info["b"]
            c = info["c"]
            alpha = info["alpha"]
            beta = info["beta"]
            gamma = info["gamma"]
            n_atoms = info["n_atoms"]
            cif_path = info["cif_path"]
        else:
            # Cell filter not applied (e.g. failed RMSD or missing CIF)
            cell_status = "not_applicable"
            cell_reason = "not_applicable"
            a = b = c = alpha = beta = gamma = np.nan
            n_atoms = np.nan
            cif_path = ""

        # ---------- Final status + failure_reason ----------
        if rmsd_status == "ok" and cif_status == "ok" and cell_status == "ok":
            final_status = "ok"
            failure_reason = "ok"
        else:
            final_status = "fail"

            # Priority of reasons:
            if rmsd_status == "fail":
                failure_reason = "rmsd_fail"
            elif rmsd_status == "not_found" and cif_status == "missing":
                # COF in plan but no rmsd entry and no CIF → likely never built / timeout
                failure_reason = "timeout_missing_cif"
            elif rmsd_status == "not_found":
                failure_reason = "no_rmsd_entry"
            elif cif_status == "missing":
                # User requested: missing CIF should be marked as timeout
                failure_reason = "timeout_missing_cif"
            elif cell_status == "fail":
                failure_reason = cell_reason
            else:
                failure_reason = "unknown"

        master_row = row.to_dict()
        master_row.update({
            "rmsd_status": rmsd_status,
            "cif_status": cif_status,
            "cell_status": cell_status,
            "final_status": final_status,
            "failure_reason": failure_reason,
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_atoms": n_atoms,
            "cif_path": cif_path,
        })

        master_rows.append(master_row)

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(MASTER_OUT, index=False)

    print("[INFO] Master status table written to:", MASTER_OUT)
    print("[INFO] Done.\n")


if __name__ == "__main__":
    main()

