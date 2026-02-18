#!/usr/bin/env python3
"""
Combine:
    - cof_master_status.csv      (RMSD + CIF + Cell filtering)
    - lammps_interface_success.csv
To produce:
    - cof_master_status_final.csv   (FULL unified status including LAMMPS stage)

LAMMPS logic:
    If cof_id appears in lammps_interface_success.csv → OK
    Else → FAIL
"""

import pandas as pd

MASTER_IN = "cof_master_status.csv"
LAMMPS_SUCC = "lammps_interface_success.csv"
FINAL_OUT = "cof_master_status_final.csv"

def main():
    # -------------------------- Load master -----------------------------
    df = pd.read_csv(MASTER_IN)
    df["cof_id"] = df["cof_id"].astype(str)

    # --------------------- Load LAMMPS success list --------------------
    try:
        df_succ = pd.read_csv(LAMMPS_SUCC)
        df_succ["cof_id"] = df_succ["cof_id"].astype(str)
        succ_ids = set(df_succ["cof_id"].tolist())
    except FileNotFoundError:
        print("[WARN] lammps_interface_success.csv not found. Assuming none successful.")
        succ_ids = set()

    # ---------------------- Determine LAMMPS status ---------------------
    lammps_status_list = []
    for cof_id in df["cof_id"]:
        if cof_id in succ_ids:
            lammps_status_list.append("ok")
        else:
            lammps_status_list.append("fail")

    df["lammps_status"] = lammps_status_list

    # ---------------------- Determine final status ----------------------
    final_status = []
    failure_reason = []

    for idx, row in df.iterrows():
        if (
            row["rmsd_status"] == "ok"
            and row["cif_status"] == "ok"
            and row["cell_status"] == "ok"
            and row["lammps_status"] == "ok"
        ):
            final_status.append("ok")
            failure_reason.append("ok")
        else:
            final_status.append("fail")
            # Priority assignment
            if row["rmsd_status"] != "ok":
                failure_reason.append("rmsd_fail")
            elif row["cif_status"] != "ok":
                failure_reason.append("missing_cif")
            elif row["cell_status"] != "ok":
                failure_reason.append("bad_cell")
            elif row["lammps_status"] != "ok":
                failure_reason.append("lammps_fail")
            else:
                failure_reason.append("unknown")

    df["final_status"] = final_status
    df["failure_reason"] = failure_reason

    # ---------------------- Save final master table ---------------------
    df.to_csv(FINAL_OUT, index=False)
    print(f"[INFO] Unified master file written to: {FINAL_OUT}")
    print("[INFO] DONE.")


if __name__ == "__main__":
    main()

