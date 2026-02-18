import os
import re
import pandas as pd

# =======================
# USER SETTINGS
# =======================
FINAL_ML = "final_cof_ml.csv"
PARENT = "COF_CIFs"                 # folder containing COF_XXXXXX subfolders
OUTCSV = "global_props_opt.csv"

# =======================
# REGEX HELPERS
# =======================
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def canon(x: str) -> str:
    return str(x).strip().upper()

# =======================
# PARSER
# =======================
def parse_res(path: str) -> dict:
    """
    Example:
      ./.../COF_000001.res    22.74772 19.63485  22.74772
    Interpreted as:
      LCD, PLD, LIFD (common Zeo++ convention)

    We keep only LCD and PLD because they are your ML targets.
    """
    with open(path, "r", errors="ignore") as f:
        nums = re.findall(NUM, f.read())
    if len(nums) >= 3:
        return {
            "LCD_A": float(nums[-3]),
            "PLD_A": float(nums[-2]),
        }
    return {}

# =======================
# MAIN
# =======================
ml = pd.read_csv(FINAL_ML)
cof_ids = [canon(x) for x in ml["cof_id"].astype(str).tolist()]

rows = []
missing_dirs = 0

for cof_id in cof_ids:
    cof_dir = os.path.join(PARENT, cof_id)
    if not os.path.isdir(cof_dir):
        missing_dirs += 1
        continue

    rec = {"cof_id": cof_id}

    res_path = os.path.join(cof_dir, f"{cof_id}.res")
    if os.path.exists(res_path):
        rec.update(parse_res(res_path))

    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUTCSV, index=False)

print("\n======================================")
print(f"✅ Global properties written → {OUTCSV}")
print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
if missing_dirs:
    print(f"⚠️ Missing COF directories (skipped): {missing_dirs}")
print("======================================\n")
