import os
import csv
import subprocess
import time

# ================== USER SETTINGS ==================

PARENT_DIR = "./cofs_for_optimization"
CSV_FILE = "./cof_preopt_summary.csv"

# Command template for LAMMPS: adjust "mpirun", "lmp" if needed
CORES_PER_JOB = 16
MAX_PARALLEL_JOBS = 3      # 3 * 16 = 48 cores

# NOTE: we will keep this template, but build the final command per job
LAMMPS_CMD_TEMPLATE = [
    "mpirun",
    "-np", str(CORES_PER_JOB),
    "lmp",           # change to "lmp_mpi" or your LAMMPS executable if needed
    "-in"            # input file will be appended per job
]

SUMMARY_OUT = "lammps_run_summary.csv"

# ====================================================


def find_lammps_input_file(folder_path: str) -> str | None:
    """
    Prefer 'in.COF' if present. Otherwise, pick one 'in.COF_*' file.
    Returns the filename (not full path) to be used with LAMMPS -in.
    """
    # 1) Prefer exact in.COF
    if os.path.exists(os.path.join(folder_path, "in.COF")):
        return "in.COF"

    # 2) Fall back to in.COF_* (including in.COF_XXXXXX)
    candidates = []
    for fn in os.listdir(folder_path):
        if fn.startswith("in.COF_") and os.path.isfile(os.path.join(folder_path, fn)):
            candidates.append(fn)

    if not candidates:
        return None

    # pick the most stable deterministic choice
    candidates.sort()
    return candidates[0]


def load_jobs_from_csv(csv_path, parent_dir):
    jobs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row.get("folder")
            cof_id = row.get("cof_id", "")
            n_atoms = row.get("n_atoms", "")

            if not folder:
                continue

            folder_path = os.path.join(parent_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"⚠️  Folder not found, skipping: {folder}")
                continue

            # Find correct in-file (in.COF or in.COF_*)
            in_filename = find_lammps_input_file(folder_path)
            if in_filename is None:
                print(f"⚠️  No in.COF or in.COF_* found in {folder}, skipping.")
                continue

            try:
                n_atoms_val = int(float(n_atoms)) if n_atoms else None
            except ValueError:
                n_atoms_val = None

            jobs.append({
                "folder": folder,
                "folder_path": folder_path,
                "cof_id": cof_id,
                "n_atoms": n_atoms_val,
                "in_filename": in_filename
            })

    # smallest first
    jobs.sort(key=lambda x: (x["n_atoms"] if x["n_atoms"] is not None else 10**12))
    return jobs


def print_progress(completed, total, running):
    bar_len = 40
    filled = int(bar_len * completed / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = (completed / total * 100) if total > 0 else 0
    status = f"[{bar}] {completed}/{total} completed | running: {running}"
    status += f" | {percent:5.1f}%"
    print("\r" + status, end="", flush=True)


def main():
    jobs = load_jobs_from_csv(CSV_FILE, PARENT_DIR)
    total_jobs = len(jobs)

    if total_jobs == 0:
        print("No jobs found. Check CSV/paths.")
        return

    print(f"Found {total_jobs} COFs to run with LAMMPS.")
    print(f"Running up to {MAX_PARALLEL_JOBS} jobs in parallel, "
          f"{CORES_PER_JOB} cores each (total {MAX_PARALLEL_JOBS * CORES_PER_JOB} cores).")

    pending = jobs[:]   # queue of jobs waiting to be launched
    running = []        # list of dicts: { 'proc', 'job', 'log_file' }
    completed_count = 0
    results = []        # for summary CSV

    print_progress(completed_count, total_jobs, len(running))

    while pending or running:
        # Launch new jobs if we have capacity
        while pending and len(running) < MAX_PARALLEL_JOBS:
            job = pending.pop(0)
            folder = job["folder"]
            folder_path = job["folder_path"]
            in_filename = job["in_filename"]

            log_path = os.path.join(folder_path, "lammps_run.log")
            log_file = open(log_path, "w")

            # Build the command *per job* so we can use in.COF_XXXXXX
            cmd = LAMMPS_CMD_TEMPLATE + [in_filename]

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=folder_path,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )

                running.append({
                    "proc": proc,
                    "job": job,
                    "log_file": log_file,
                    "log_path": log_path,
                    "cmd": cmd
                })

                print(f"\n▶ Launched: {folder}  (input: {in_filename})")
            except Exception as e:
                print(f"\n❌ Failed to launch {folder}: {e}")
                try:
                    log_file.write(f"Failed to launch job: {e}\n")
                finally:
                    log_file.close()

                results.append({
                    "folder": folder,
                    "cof_id": job["cof_id"],
                    "n_atoms": job["n_atoms"],
                    "in_file": in_filename,
                    "status": "launch_failed",
                    "return_code": "",
                    "log_file": log_path
                })

        # Check running jobs for completion
        still_running = []
        for r in running:
            proc = r["proc"]
            job = r["job"]
            log_file = r["log_file"]
            log_path = r["log_path"]

            ret = proc.poll()
            if ret is None:
                still_running.append(r)
                continue

            # Process finished
            log_file.close()
            completed_count += 1

            status = "success" if ret == 0 else "error"
            if status == "error":
                print(f"\n❌ Job failed: {job['folder']} (return code {ret})")
            else:
                print(f"\n✓ Job completed: {job['folder']}")

            results.append({
                "folder": job["folder"],
                "cof_id": job["cof_id"],
                "n_atoms": job["n_atoms"],
                "in_file": job["in_filename"],
                "status": status,
                "return_code": ret,
                "log_file": log_path
            })

            print_progress(completed_count, total_jobs, len(still_running))

        running = still_running
        time.sleep(2)

    print("\n\nAll jobs finished.")

    # Write summary CSV
    with open(SUMMARY_OUT, "w", newline="") as f:
        fieldnames = ["folder", "cof_id", "n_atoms", "in_file", "status", "return_code", "log_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Summary written to: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
