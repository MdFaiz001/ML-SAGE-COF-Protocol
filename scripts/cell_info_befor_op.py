import os
import math
import csv

PARENT = "./cofs_for_optimization"
OUTPUT = "cof_preopt_summary.csv"

def parse_data_file(path):
    """Parse a LAMMPS data file and extract box parameters."""
    n_atoms = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0  # default tilt values
    
    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # number of atoms
        if "atoms" in line and n_atoms is None:
            try:
                n_atoms = int(line.split()[0])
            except:
                pass

        # box bounds
        if "xlo xhi" in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])

        if "ylo yhi" in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])

        if "zlo zhi" in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])

        # tilt factors (triclinic systems)
        if "xy xz yz" in line:
            parts = line.split()
            xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])

    # compute Lx, Ly, Lz
    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # lattice vector lengths (LAMMPS triclinic vectors)
    a = Lx
    b = math.sqrt(xy**2 + Ly**2)
    c = math.sqrt(xz**2 + yz**2 + Lz**2)

    # angles
    # cos(gamma) = (a·b)/(|a||b|)
    cos_gamma = (Lx * xy) / (a * b)
    # cos(beta) = (a·c)/(|a||c|)
    cos_beta = (Lx * xz) / (a * c)
    # cos(alpha) = (b·c)/(|b||c|)
    cos_alpha = (xy * xz + Ly * yz) / (b * c)

    # convert to degrees safely
    alpha = math.degrees(math.acos(max(min(cos_alpha, 1), -1)))
    beta  = math.degrees(math.acos(max(min(cos_beta, 1), -1)))
    gamma = math.degrees(math.acos(max(min(cos_gamma, 1), -1)))

    return n_atoms, a, b, c, alpha, beta, gamma


# ===========================
# MAIN SCRIPT
# ===========================

rows = []
rows.append(["folder", "cof_id", "n_atoms", "a", "b", "c", "alpha", "beta", "gamma"])

for folder in os.listdir(PARENT):
    folder_path = os.path.join(PARENT, folder)
    if not os.path.isdir(folder_path):
        continue
    if "__COF_" not in folder:
        continue

    # example: bto__COF_000123 → cof_id = COF_000123
    cof_id = folder.split("__")[-1]

    data_filename = f"data.{cof_id}"

    data_path = os.path.join(folder_path, data_filename)
    if not os.path.exists(data_path):
        print(f"⚠️  MISSING data file in {folder}")
        continue

    try:
        n_atoms, a, b, c, alpha, beta, gamma = parse_data_file(data_path)

        rows.append([
            folder,
            cof_id,
            n_atoms,
            round(a, 6),
            round(b, 6),
            round(c, 6),
            round(alpha, 6),
            round(beta, 6),
            round(gamma, 6)
        ])

        print(f"✓ Processed {folder}")

    except Exception as e:
        print(f"❌ Error processing {folder}: {e}")

# write summary CSV
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("\n===================================")
print(f"Summary written to: {OUTPUT}")
print("===================================\n")

