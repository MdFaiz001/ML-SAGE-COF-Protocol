# `global_prop_ext.py` — Extract global Zeo++ pore descriptors (LCD, PLD) from `.res` files → `global_props_opt.csv`

## Purpose

After you run Zeo++ network analysis (which creates one `.res` file per COF), this script collects the pore descriptors required for ML and GA screening—specifically:

* **LCD_A**: Largest Cavity Diameter (Å)
* **PLD_A**: Pore Limiting Diameter (Å)

It does this by:

1. reading a list of target COFs from `final_cof_ml.csv`
2. locating each COF’s `.res` file in `COF_CIFs/COF_XXXXXX/`
3. parsing numeric values from the `.res` file
4. writing a compact output table: `global_props_opt.csv`

---

## Inputs required

### 1) List of COFs to extract properties for

```python
FINAL_ML = "final_cof_ml.csv"
```

This CSV must contain at least a column:

* `cof_id`

Important: `cof_id` is canonicalized using:

```python
canon(x) = str(x).strip().upper()
```

So expected IDs are like:

* `COF_000001`
* `COF_007370`

### 2) Optimized CIF/RES folder tree

```python
PARENT = "COF_CIFs"
```

Expected structure for each COF id:

```text
COF_CIFs/
└─ COF_000001/
   ├─ COF_000001.cif
   └─ COF_000001.res   (required for extraction)
```

The `.res` file must already exist (produced by Zeo++ `network -res`).

---

## Output produced

### `global_props_opt.csv`

```python
OUTCSV = "global_props_opt.csv"
```

This CSV contains:

* `cof_id`
* `LCD_A` (if `.res` exists and parse succeeds)
* `PLD_A` (if `.res` exists and parse succeeds)

If `.res` is missing or parsing fails, the row may contain only `cof_id` with empty property columns.

---

## Parsing logic (critical)

### Numeric pattern

The script uses a robust floating-number regex:

```python
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
```

So it captures:

* integers: `12`
* floats: `12.34`
* scientific notation: `1.2e+02`

### How `parse_res()` works

It reads the entire `.res` file as text and extracts **all numbers** using regex:

```python
nums = re.findall(NUM, f.read())
```

Then it interprets the **last three numbers** as:

```text
...  LCD   PLD   LIFD
```

and returns:

* `LCD_A = float(nums[-3])`
* `PLD_A = float(nums[-2])`

**Important assumption:**
This assumes your Zeo++ `.res` file ends with three key values in the conventional order (LCD, PLD, LIFD). The script intentionally keeps only LCD and PLD because those are your ML targets. 

If fewer than 3 numbers are found:

* returns `{}` (no properties for that COF)

---

## Main execution logic (step-by-step)

### Step 1 — Load `final_cof_ml.csv`

```python
ml = pd.read_csv(FINAL_ML)
cof_ids = [canon(x) for x in ml["cof_id"].astype(str).tolist()]
```

So the script iterates only over COFs in this final list.

### Step 2 — For each COF id, check directory exists

Expected directory:

```python
cof_dir = os.path.join(PARENT, cof_id)
```

If missing:

* increments `missing_dirs`
* skips that COF completely

### Step 3 — Try to parse `.res`

Expected `.res` path:

```python
res_path = os.path.join(cof_dir, f"{cof_id}.res")
```

If it exists:

* parse it
* add keys `LCD_A`, `PLD_A` into the record

If it does not exist:

* record remains only `{"cof_id": cof_id}`
* still included in output rows

### Step 4 — Write results

All records are written to:

```python
df.to_csv(OUTCSV, index=False)
```

### Step 5 — Console summary

At the end it prints:

* output filename
* number of rows
* number of columns
* how many COF directories were missing and skipped

---

## What this script does NOT do

* It does not run Zeo++ itself (that is `run_zeo_op_global.py`)
* It does not validate that the `.res` file corresponds to the same CIF version
* It does not compute any other Zeo++ descriptors beyond LCD/PLD
* It does not enforce completeness (missing `.res` will still produce an output row with only cof_id)

---

## Expected folder structure

```text
project_root/
├─ final_cof_ml.csv
├─ COF_CIFs/
│  └─ COF_000001/
│     ├─ COF_000001.cif
│     └─ COF_000001.res
├─ global_prop_ext.py
└─ global_props_opt.csv     (generated)
```

---

## How to run

From the project root:

```bash
python global_prop_ext.py
```

---

## Notes for thesis / paper 

* “Global pore descriptors (LCD and PLD) were extracted from Zeo++ network analysis outputs (`.res`) using a dedicated parser that reads the terminal triplet of Zeo++ summary values and retains LCD and PLD in Å for downstream modeling.” 

---

