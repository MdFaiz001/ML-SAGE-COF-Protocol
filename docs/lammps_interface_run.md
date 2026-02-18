# `prepare_lammps_inputs_resume_live.py` — Resumable parallel `lammps-interface` runner + live logs + live copy to optimization folder

## Purpose

This script is the **bridge between generated COF CIFs** (after filtering) and **LAMMPS geometry optimization**. It automates:

* Running `lammps-interface --minimize` on each COF CIF to generate:

  * `data.COF_XXXXXX`
  * `in.COF_XXXXXX`
* Running jobs **in parallel** using `ProcessPoolExecutor` (full CPU utilization)
* **Sorting by `n_atoms`** (smallest first) for higher throughput
* **Resume support**: safely re-run anytime, it skips already completed COFs
* **Live logging**: results are appended to CSV logs immediately after each job finishes
* **Live copying**: successful COFs are immediately copied into the optimization staging folder
* **Template-based `in.COF` rewrite**: creates a final `in.COF` ready for your LAMMPS optimization pipeline

---

## Required folder structure and inputs

### Input folders and files (must exist)

1. **`filtered_cofs_ready/`**  *(input root for COFs ready to run)*

   * Contains subfolders for each COF, typically named:

     * `<topology>__COF_XXXXXX`
   * Each subfolder must contain a `.cif` file (any name, but must end with `.cif`)

2. **`final_cofs_after_rmsd_and_cell_filter.csv`** *(the run list)*
   Must include columns:

   * `cof_id` (format: `COF_000123`)
   * `output_dir` (folder name inside `filtered_cofs_ready/`, e.g., `dia__COF_000123`)
   * optional: `n_atoms` (if present, used to sort small → large)

3. **`in.COF`** *(LAMMPS template input file)*

   * Used as a template
   * Script rewrites `read_data` and `write_data` lines automatically

### Output folders and files (created/updated)

1. **`cofs_for_optimization/`** *(optimization staging folder)*

   * Created if missing
   * Will contain copies of successful COF folders
   * Each copied folder includes:

     * `data.COF_XXXXXX`
     * original `*.cif`
     * and a rewritten **final** `in.COF` (template-based)

2. **Live log CSV files (created/updated as jobs finish)**

* `lammps_interface_log.csv` — all results (OK + failures)
* `lammps_interface_errors.csv` — only non-OK results
* `lammps_interface_success.csv` — only OK results

---

## User settings (edit at top of file)

* `COF_ROOT = "filtered_cofs_ready"`

* `FINAL_CSV = "final_cofs_after_rmsd_and_cell_filter.csv"`

* `FORCE_FIELD = "Dreiding"`
  (can be `"UFF"` etc., passed to `lammps-interface`)

* `NPROCS = 44`
  Number of parallel workers (match your machine core availability)

* `TEMPLATE_INCOF = "./in.COF"`
  Template LAMMPS input

* `OPT_ROOT = "cofs_for_optimization"`
  Output staging folder

* `LOG_CSV, ERR_CSV, SUCCESS_CSV`
  Names of log files

---

## High-level workflow (what happens when you run it)

### Step 1 — Load template `in.COF`

At startup, the script reads `./in.COF` into memory (`TEMPLATE_LINES`).

* If `in.COF` is missing: the script stops with:
  `FileNotFoundError: Template in.COF not found`

### Step 2 — Load final COF list from CSV

Reads `final_cofs_after_rmsd_and_cell_filter.csv`.

* Forces `cof_id` and `output_dir` to string
* If `n_atoms` exists: sorts by `n_atoms` ascending (smallest first)

### Step 3 — Determine which COFs are already complete (Resume logic)

Resume support is implemented via **two independent success detectors**, then unioned:

#### A) Previous success log (`lammps_interface_success.csv`)

If it exists:

* loads it and reads list of successful `cof_id`s → `success_ids_prev`

#### B) File-based success detection (inside `filtered_cofs_ready/`)

For every COF in the final CSV:

* checks whether BOTH files already exist inside its folder:

**Expected filenames are derived from cof_id:**

If:

```
cof_id = COF_000123
cof_num = 000123
```

Then expected outputs are:

* `data_name  = data.COF_000123`
* `incof_name = in.COF_000123`

The COF is considered already successful if both exist:

* `filtered_cofs_ready/<output_dir>/data.COF_000123`
* `filtered_cofs_ready/<output_dir>/in.COF_000123`

These are collected into `file_success_ids`.

#### Final resume set

```text
success_ids_initial = success_ids_prev ∪ file_success_ids
```

### Step 4 — Build job list (skipping completed)

Any `cof_id` in `success_ids_initial` is skipped; all others become jobs.

Each job dict contains:

* `cof_id`
* `folder` (= output_dir)
* `n_atoms` (if present)

### Step 5 — Ensure optimization folder exists

Creates `cofs_for_optimization/` if missing.

### Step 6 — Live copy “already-successful” COFs into OPT_ROOT

Before running new jobs, the script **also copies all known successful COFs** into `cofs_for_optimization/`, so OPT_ROOT stays consistent.

This copy is **overwrite mode**:

* if destination folder already exists, it is deleted and copied fresh.

### Step 7 — Run remaining jobs in parallel (`ProcessPoolExecutor`)

For remaining jobs only, the script launches `run_job(job)` in parallel.

Progress bar shown via tqdm:

* `desc="Running lammps-interface"`

---

## Worker logic: `run_job(job)` (exact status rules)

For each COF:

### A) Folder existence check

If `filtered_cofs_ready/<folder>` does not exist:

* status = `NO_FOLDER`
* error = `"Folder not found: ..."`
* has_data = False
* has_incof = False
* time_sec = 0

### B) CIF existence check

It finds the first `.cif` in the folder (`find_cif()`).

If no CIF found:

* status = `NO_CIF`
* error = `"No CIF file in folder"`
* has_data = False
* has_incof = False
* time_sec = 0

### C) Runs lammps-interface

Command executed (inside that COF folder):

```bash
lammps-interface --force_field Dreiding --minimize <cif_filename>
```

* `cwd = folder_path` (important: outputs appear inside COF folder)
* captures stdout/stderr in memory
* measures runtime in seconds (`time_sec`)

If Python raises an exception (e.g., executable missing):

* status = `EXCEPTION`
* error = exception string
* has_data/has_incof False
* time_sec = 0

### D) Output file validation (critical)

Even if `lammps-interface` returns code 0, the script still checks that both output files exist:

Expected output filenames are always:

* `data.COF_XXXXXX`
* `in.COF_XXXXXX`

These are computed from `cof_id`, **not from CIF name**.

If both files exist AND returncode == 0:

* status = `OK`
* error = ""

Otherwise:

* status = `FAILED`
* error = `stderr` (captured from lammps-interface)
* has_data / has_incof reflect actual file existence

So **FAILED** happens when:

* return code is non-zero OR
* data file missing OR
* in.COF_XXXXXX missing

---

## Live logging behavior (what goes into each CSV)

After each job finishes, the result dict is appended to CSVs.

Every result dict contains:

* `cof_id`
* `folder`
* `n_atoms`
* `status` (OK / FAILED / NO_FOLDER / NO_CIF / EXCEPTION)
* `error` (stderr or message)
* `has_data` (True/False)
* `has_incof` (True/False)
* `time_sec` (elapsed runtime)

### Logs:

1. `lammps_interface_log.csv`

   * receives **every** result

2. `lammps_interface_success.csv`

   * receives only `status == "OK"`

3. `lammps_interface_errors.csv`

   * receives everything else (FAILED/NO_FOLDER/NO_CIF/EXCEPTION)

> Logging is **live** (append per completion), so stopping the script mid-run still preserves completed logs.

---

## Live copying + final input preparation (for OK jobs)

For each `OK` job, it immediately:

### A) Copies folder to `cofs_for_optimization/`

Function: `copy_and_prepare_opt_folder(cof_id, folder)`

* Source: `filtered_cofs_ready/<folder>`
* Destination: `cofs_for_optimization/<folder>`
* If destination exists: it is deleted and overwritten

### B) Writes a new final `in.COF` inside destination

Function: `write_new_incof(dest_folder, cof_id)`

It creates:

* `cofs_for_optimization/<folder>/in.COF`

by taking the template `./in.COF` and replacing:

* any line containing `read_data` → becomes:

  ```
  read_data       data.COF_XXXXXX
  ```

* any line containing `write_data` → becomes:

  ```
  write_data      relaxed_COF_XXXXXX.data
  ```

It also removes the old `in.COF_XXXXXX` if present in dest folder.

**Key outputs in OPT folder per successful COF:**

* `data.COF_XXXXXX`  *(from lammps-interface)*
* `in.COF`           *(rewritten final input, ready for LAMMPS run)*
* `relaxed_COF_XXXXXX.data` *(will be produced later by LAMMPS run)*

---

## End-of-run summary printed

At completion, it prints a summary:

* Total COFs in CSV
* Previously successful
* Newly successful this run
* Total successful overall
* Locations of three log CSV files
* Confirmation that OPT_ROOT contains all successful COFs

---

## Expected directory layout (after running)

```text
filtered_cofs_ready/
└─ dia__COF_000123/
   ├─ COF_000123.cif
   ├─ data.COF_000123          (generated by lammps-interface)
   └─ in.COF_000123            (generated by lammps-interface)

cofs_for_optimization/
└─ dia__COF_000123/
   ├─ COF_000123.cif
   ├─ data.COF_000123
   └─ in.COF                   (rewritten from template)

lammps_interface_log.csv
lammps_interface_success.csv
lammps_interface_errors.csv
```

---

## How to run

From the project root containing the CSV and folders:

```bash
python prepare_lammps_inputs_resume_live.py
```

---



1. **Success definition is strict**

   * A COF is counted successful only when both `data.COF_XXXXXX` and `in.COF_XXXXXX` exist and return code is zero.

2. **Resume behavior**

   * Resuming is based on:

     * prior success log OR
     * existing output files in `filtered_cofs_ready/`
   * Hence it is safe to stop & rerun anytime.

3. **Naming convention is COF-ID-driven**

   * Output filenames are derived from `cof_id` not the CIF filename.
   * This ensures consistent matching across pipeline stages.

4. **Template-based final `in.COF`**

   * Ensures all subsequent optimizations use unified settings while referencing per-COF data filenames.

