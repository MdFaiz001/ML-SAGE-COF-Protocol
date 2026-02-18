import os
import re
import pandas as pd

# -----------------------------
# INPUTS
# -----------------------------
META_IN = "cof_meta_model1_clean.csv"

# Folder that contains COF subfolders, each having:
#   COF_XXXXXX_FLP_large_summary.txt
ROOT_DIR = "COF_CIFs"   # change if needed

RADIUS_TAG = "large"   # "small" | "medium" | "large"

# -----------------------------
# OUTPUT
# -----------------------------
OUT_META = f"cof_meta_model2_flp_{RADIUS_TAG}.csv"

# -----------------------------
# Helpers
# -----------------------------
def parse_summary(path):
    """
    Parse summary.txt written by save_outputs().

    Expected lines:
      N_base_FLP      = 22
      C_FLP           = 8.854163
      N_base_FLP_sim  = 8
      C_FLP_sim       = 4.045839
    """
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab_int(key):
        m = re.search(rf"{re.escape(key)}\s*=\s*([0-9]+)", txt)
        return int(m.group(1)) if m else None

    def grab_float(key):
        m = re.search(rf"{re.escape(key)}\s*=\s*([0-9]*\.?[0-9]+)", txt)
        return float(m.group(1)) if m else None

    return {
        "N_base_FLP": grab_int("N_base_FLP"),
        "C_FLP": grab_float("C_FLP"),
        "N_base_FLP_sim": grab_int("N_base_FLP_sim"),
        "C_FLP_sim": grab_float("C_FLP_sim"),
    }

# -----------------------------
# Main
# -----------------------------
print("Loading meta...")
meta = pd.read_csv(META_IN)
print("  COFs in meta:", len(meta))

rows = []
missing = 0

for cof_id in meta["cof_id"].astype(str).values:
    cof_dir = os.path.join(ROOT_DIR, cof_id)
    summary = os.path.join(cof_dir, f"{cof_id}_FLP_{RADIUS_TAG}_summary.txt")

    if not os.path.exists(summary):
        missing += 1
        continue

    d = parse_summary(summary)
    d["cof_id"] = cof_id
    rows.append(d)

targets = pd.DataFrame(rows)
print("Parsed targets:", len(targets))
print("Missing summaries:", missing)

# Merge
df = meta.merge(targets, on="cof_id", how="inner")
print("After merge:", len(df))

# Choose ONE target for Model-2:
# Recommended: C_FLP_sim
TARGET_COL = "C_FLP_sim"
df = df[df[TARGET_COL].notna()].copy()
df.reset_index(drop=True, inplace=True)

df.to_csv(OUT_META, index=False)

print("\n✅ Model-2 meta created")
print("Output:", OUT_META)
print("Final COFs:", len(df))
print("Target:", TARGET_COL, "range:", f"{df[TARGET_COL].min():.6f} – {df[TARGET_COL].max():.6f}")
