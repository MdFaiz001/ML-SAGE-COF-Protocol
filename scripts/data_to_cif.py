#!/usr/bin/env python3
import os
import re
import csv
from ase.io import read, write

# ================================
# USER SETTINGS
# ================================
PARENT_DIR = "./cofs_for_optimization"        # folder containing subdirectories
OUT_DIR    = "./COF_CIFs"                   # final cif directory
DATA_REGEX = re.compile(r"relaxed_COF_(\d+)\.data$", re.IGNORECASE)

# create out directory
os.makedirs(OUT_DIR, exist_ok=True)

log_rows = []

for root, dirs, files in os.walk(PARENT_DIR):
    for file in files:
        match = DATA_REGEX.match(file)
        if not match:
            continue
        
        cof_id = match.group(1)  # e.g. '007370'
        data_path = os.path.join(root, file)

        # subdirectory for this COF in output
        cof_out_dir = os.path.join(OUT_DIR, f"COF_{cof_id}")
        os.makedirs(cof_out_dir, exist_ok=True)

        cif_file = os.path.join(cof_out_dir, f"COF_{cof_id}.cif")

        print(f"Converting {data_path} -> {cif_file}")

        status = "ok"
        msg    = ""

        try:
            atoms = read(data_path, format="lammps-data", style="full")
            atoms.set_pbc((True, True, True))   # ensure periodic boundaries
            write(cif_file, atoms, format="cif")

        except Exception as e:
            status = "error"
            msg = str(e)

        log_rows.append({
            "cof_id"   : cof_id,
            "data_path": data_path,
            "cif_path" : cif_file,
            "status"   : status,
            "message"  : msg
        })

# logging output
log_csv = os.path.join(OUT_DIR, "conversion_summary.csv")
with open(log_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["cof_id","data_path","cif_path","status","message"])
    writer.writeheader()
    writer.writerows(log_rows)

print("\n✔ Conversion complete!")
print(f"→ CIFs stored inside : {OUT_DIR}")
print(f"→ Log saved at      : {log_csv}")

