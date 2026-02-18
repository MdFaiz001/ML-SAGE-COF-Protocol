# `launch_lammps_cofs.py` — Parallel LAMMPS optimizer launcher (MPI) + per-COF log files + run summary CSV

## Purpose

This script launches **many independent LAMMPS geometry-optimization runs** for the COFs that are already prepared in `cofs_for_optimization/` (i.e., each COF folder contains a valid LAMMPS input file and a data file). It is designed to:

* Read a job list from a CSV (`cof_preopt_summary.csv`)
* Launch **multiple LAMMPS MPI jobs in parallel**
* Allocate a fixed number of cores per job (`CORES_PER_JOB`)
* Keep a fixed maximum number of concurrent jobs (`MAX_PARALLEL_JOBS`)
* Write a **per-COF LAMMPS log** into each COF folder
* Write a global summary file `lammps_run_summary.csv` capturing job status and log paths

---

## Inputs required

### 1) Parent directory of COF job folders

```python
PARENT_DIR = "./cofs_for_optimization"
```

Expected structure:

```
cofs_for_optimization/
└─ <folder>/
   ├─ data.COF_XXXXXX          (required by LAMMPS in-file)
   ├─ in.COF                   (preferred) OR in.COF_XXXXXX (fallback)
   └─ (other files)
```

### 2) Job list CSV

```python
CSV_FILE = "./cof_preopt_summary.csv"
```

This CSV must have (at minimum) the column:

* `folder`  → folder name inside `cofs_for_optimization/`

Optional columns used if present:

* `cof_id`
* `n_atoms` (used for sorting jobs smallest-first)

---

## Core settings you control

### Parallelization knobs

```python
CORES_PER_JOB = 16
MAX_PARALLEL_JOBS = 3      # 3 * 16 = 48 cores
```

So total core usage ≈ `CORES_PER_JOB × MAX_PARALLEL_JOBS`.

### LAMMPS command template

```python
LAMMPS_CMD_TEMPLATE = ["mpirun", "-np", str(CORES_PER_JOB), "lmp", "-in"]
```

This means each job is launched as:

```bash
mpirun -np 16 lmp -in <input_file>
```

You may need to change `"lmp"` to your cluster’s LAMMPS executable, e.g. `lmp_mpi`.

### Output summary file

```python
SUMMARY_OUT = "lammps_run_summary.csv"
```

---

## How the script selects the LAMMPS input file

Function: `find_lammps_input_file(folder_path)`

Priority:

1. If `in.COF` exists → use it
2. Else search for `in.COF_*` (including `in.COF_000123`)

   * candidates are sorted alphabetically
   * the first one is selected deterministically

If neither exists → job is skipped.

**Why this matters:**
Your earlier pipeline writes a unified `in.COF` for optimization, but if not present, this launcher still works using the original `in.COF_XXXXXX`.

---

## Job loading and ordering

Function: `load_jobs_from_csv(csv_path, parent_dir)`

For each row in CSV:

* reads `folder`, `cof_id`, `n_atoms`
* verifies folder exists: `cofs_for_optimization/<folder>`
* verifies a valid input file exists via `find_lammps_input_file`
* stores a job dict with:

  * `folder`, `folder_path`, `cof_id`, `n_atoms`, `in_filename`

Then it sorts jobs by **increasing n_atoms**:

* small systems run first for throughput

If `n_atoms` is missing/invalid, it is treated as very large and pushed to the end.

---

## Execution model (how parallel running works)

The script uses a **queue scheduler**:

* `pending` = list of jobs not yet started
* `running` = list of active processes (each with `Popen` handle + metadata)
* It continuously:

  1. launches new jobs until `len(running) == MAX_PARALLEL_JOBS`
  2. checks for finished jobs using `proc.poll()`
  3. records completion status
  4. sleeps 2 seconds and repeats until all done

This is a lightweight internal job manager.

---

## What each job produces

### Per-job log file

For each COF folder, a file is created:

```
cofs_for_optimization/<folder>/lammps_run.log
```

Stdout and stderr are redirected into this log:

```python
stdout=log_file, stderr=subprocess.STDOUT
```

So **all LAMMPS output is inside that COF folder**, making debugging localized.

---

## Status classification (important)

When a job is launched:

* If launch fails (e.g., mpirun or lmp missing), it records:

  * `status = "launch_failed"`

When the job completes:

* `return_code = 0` → `status = "success"`
* `return_code != 0` → `status = "error"`

This is purely based on process exit code.

---

## Global output: `lammps_run_summary.csv`

At the end, the script writes a summary CSV with columns:

* `folder`
* `cof_id`
* `n_atoms`
* `in_file`  (which input file was used)
* `status`   (`success`, `error`, `launch_failed`)
* `return_code`
* `log_file` (path to `lammps_run.log` inside the folder)

This file is the **single source of truth** for:

* which COFs finished
* which failed
* where to check logs

---

## Progress display (console)

The script prints:

* Total number of jobs found
* Parallel configuration (jobs × cores)
* A live progress bar:

  * `completed/total` and `running: N`
* On each launch:

  * `▶ Launched: <folder> (input: <in_filename>)`
* On completion:

  * `✓ Job completed: <folder>`
  * or `❌ Job failed: <folder> (return code ...)`

---

## What this script does NOT do 

* It does **not** generate `cof_preopt_summary.csv` (that must exist already)
* It does **not** check whether LAMMPS produced the expected relaxed output file

  * It only checks the exit code and writes logs
* It does **not** implement resume/skip logic by detecting already-relaxed files

  * If you re-run it, it will re-launch jobs again unless you manually filter CSV

(If you want, we can upgrade it later to be fully resumable by checking presence of `relaxed_COF_XXXXXX.data` before launching.)

---

## Expected folder structure

```text
cofs_for_optimization/
└─ dia__COF_000123/
   ├─ data.COF_000123
   ├─ in.COF
   └─ lammps_run.log          (created by this script)

cof_preopt_summary.csv        (input)
lammps_run_summary.csv        (output)
```

---

## How to run

From the project root:

```bash
python launch_lammps_cofs.py
```

---

