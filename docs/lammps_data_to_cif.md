

# `data_to_cif.py` — Convert LAMMPS relaxed `*.data` → periodic CIFs (ASE) + conversion_summary.csv

## Purpose

After your LAMMPS geometry optimization runs finish, each COF folder contains an optimized structure written as a LAMMPS data file:

* `relaxed_COF_XXXXXX.data`

This script scans **all subdirectories** inside `cofs_for_optimization/`, finds those relaxed `.data` files, converts each into a **CIF file** using **ASE**, and writes all CIFs into a clean output collection directory (`COF_CIFs/`) with a single master conversion log.

This is useful for:

* saving optimized structures in a standard crystallographic format
* downstream tools that prefer CIF (Zeo++, visualization, structure archiving)
* building a final dataset of optimized COFs for publication / ML / pore analysis

---

## Inputs required

### 1) Parent directory containing optimized COF folders

```python
PARENT_DIR = "./cofs_for_optimization"
```

This folder is expected to contain many subfolders (one per COF), and those folders may contain:

* `relaxed_COF_XXXXXX.data`  ✅ (required for conversion)
* other LAMMPS outputs/logs

The script searches recursively (`os.walk`), so the relaxed data file can be in any nested subfolder under `PARENT_DIR`.

### 2) Naming convention (strict)

The script only converts files matching this regex:

```python
DATA_REGEX = r"relaxed_COF_(\d+)\.data$"
```

Examples that **will match**:

* `relaxed_COF_007370.data`
* `relaxed_COF_12.data`

Examples that **will not match**:

* `relaxed_COF_007370.DATA` (uppercase extension might fail depending on filesystem, but regex uses IGNORECASE so extension case is fine)
* `relaxed_COF_007370.data.gz`
* `relaxed_GA_base7_00389.data`

So your LAMMPS optimization stage must output in the exact format:
`relaxed_COF_<digits>.data`

---

## Outputs produced

### 1) Output CIF directory

```python
OUT_DIR = "./COF_CIFs"
```

For each converted COF, the script creates:

```
COF_CIFs/
└─ COF_<id>/
   └─ COF_<id>.cif
```

Example:

```
COF_CIFs/
└─ COF_007370/
   └─ COF_007370.cif
```

This organization keeps CIFs separated by COF id, avoiding filename collisions and making it easy to package/share.

### 2) Conversion log (very important)

At the end it writes:

```
COF_CIFs/conversion_summary.csv
```

Each row contains:

* `cof_id` — the digits captured from filename (e.g., `007370`)
* `data_path` — full path to the `.data` file found
* `cif_path` — output CIF path written
* `status` — `ok` or `error`
* `message` — exception message if conversion failed

This CSV is your accountability record: how many converted successfully, which failed, and why.

---

## Exact step-by-step logic

### Step 1 — Create output folder

```python
os.makedirs(OUT_DIR, exist_ok=True)
```

Ensures `COF_CIFs/` exists.

### Step 2 — Recursive scan

```python
for root, dirs, files in os.walk(PARENT_DIR):
```

Walks through every subdirectory under `cofs_for_optimization/`.

### Step 3 — Select only relaxed data files

For each file, apply:

```python
match = DATA_REGEX.match(file)
```

Only filenames matching `relaxed_COF_(\d+).data` are processed.

### Step 4 — Extract COF id from filename

```python
cof_id = match.group(1)
```

This is the numeric part of the filename.

### Step 5 — Create a COF-specific output directory

```python
cof_out_dir = os.path.join(OUT_DIR, f"COF_{cof_id}")
os.makedirs(cof_out_dir, exist_ok=True)
```

### Step 6 — Convert using ASE (core conversion)

Conversion logic:

```python
atoms = read(data_path, format="lammps-data", style="full")
atoms.set_pbc((True, True, True))
write(cif_file, atoms, format="cif")
```

Important points:

* `format="lammps-data", style="full"`
  Means the `.data` file must be readable as **LAMMPS “full” atom_style** data.
* `set_pbc(True,True,True)`
  Forces periodic boundary conditions in the output CIF (crucial for crystalline COFs).

### Step 7 — Error handling

If ASE fails to read or write:

* status becomes `error`
* message stores the exception text
* the script continues converting other COFs (no crash)

### Step 8 — Save summary CSV

All conversion results are written to:

```python
COF_CIFs/conversion_summary.csv
```

---

## Console output

For each successful attempt, it prints a live line like:

```
Converting <data_path> -> <cif_path>
```

At the end, it prints:

* “Conversion complete!”
* output directory location
* log CSV location

---

## Expected folder structure

Before running:

```text
cofs_for_optimization/
└─ dia__COF_007370/
   ├─ in.COF
   ├─ data.COF_007370
   ├─ relaxed_COF_007370.data   (must exist)
   └─ lammps_run.log
```

After running:

```text
COF_CIFs/
└─ COF_007370/
   └─ COF_007370.cif
COF_CIFs/conversion_summary.csv
```

---

## How to run

From project root:

```bash
python data_to_cif.py
```

---

## Notes / limitations 

1. **Only converts files that match the relaxed filename pattern**
   If your LAMMPS run writes a different naming scheme, adjust `DATA_REGEX`.

2. **Assumes LAMMPS data is “full” style**
   If your LAMMPS output uses another style (atomic, charge, molecular), the ASE read may fail unless updated.

3. **Does not verify chemical correctness**
   It converts geometry + cell; it does not validate bonding, charges, or atom typing.

---

