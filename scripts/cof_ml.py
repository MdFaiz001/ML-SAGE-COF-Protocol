import pandas as pd

# --------------------------------------------------
# Input files
# --------------------------------------------------
cof_master_file = "cof_master_status_final.csv"
lammps_summary_file = "lammps_run_summary.csv"

# --------------------------------------------------
# Output file
# --------------------------------------------------
output_file = "final_cof_ml.csv"

# --------------------------------------------------
# Correct COF identifier (verified from CSV)
# --------------------------------------------------
COF_ID_COL = "cof_id"

# --------------------------------------------------
# Read input CSVs
# --------------------------------------------------
cof_master_df = pd.read_csv(cof_master_file)
lammps_df = pd.read_csv(lammps_summary_file)

# --------------------------------------------------
# 1. Filter master → only final_status == OK
# --------------------------------------------------
cof_master_ok = cof_master_df[
    cof_master_df["final_status"].astype(str).str.strip() == "ok"
].copy()

print(f"[INFO] COFs with final_status == OK: {len(cof_master_ok)}")

# --------------------------------------------------
# 2. Extract COF IDs from LAMMPS summary
# --------------------------------------------------
lammps_cof_ids = set(
    lammps_df[COF_ID_COL].astype(str).str.strip()
)

print(f"[INFO] COFs present in LAMMPS summary: {len(lammps_cof_ids)}")

# --------------------------------------------------
# 3. Intersection: OK + LAMMPS-success
# --------------------------------------------------
final_ml_df = cof_master_ok[
    cof_master_ok[COF_ID_COL].astype(str).str.strip().isin(lammps_cof_ids)
].copy()

print(f"[INFO] Final COFs selected for ML universe: {len(final_ml_df)}")

# --------------------------------------------------
# 4. Write ML-ready metadata file
# --------------------------------------------------
final_ml_df.to_csv(output_file, index=False)

print(f"\n[SUCCESS] ML universe file written → {output_file}")

