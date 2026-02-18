#!/usr/bin/env python3

"""
ALL-IN-ONE FLP BATCH RUNNER (FULLY RESUMABLE VERSION)

✔ Automatically runs Zeo++ network only if nt2 missing
✔ Computes FLP pockets only for missing radii
✔ Skips COFs that already have all results
✔ JSON format matches single-COF notebook
✔ Safe to stop & restart anytime
"""

import os, subprocess, json
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from ase.io import read as ase_read

try:
    from tqdm import tqdm
    TQDM = True
except:
    TQDM = False


# ==========================================================
# USER SETTINGS
# ==========================================================

ROOT_DIR = "COF_CIFs"
MAX_WORKERS = 48

ACID_SIZES = {
    "small"  : 2.2,
    "medium" : 3.8,
    "large"  : 4.5
}

DEFAULT_R_CO2      = 1.65
DEFAULT_R_H2       = 1.20
DEFAULT_CLEARANCE  = 1.5
DEFAULT_D_MIN      = 2.0
DEFAULT_D_MAX      = 5.0
DEFAULT_THETA_MAX  = 75.0
DEFAULT_LAMBDA_CO2 = 3.0
DEFAULT_LAMBDA_H2  = 3.0
DEFAULT_POCKET_BUF = 0.7


# ==========================================================
@dataclass
class VoronoiNode:
    idx: int
    pos: np.ndarray
    radius: float

@dataclass
class FLPSite:
    base_index: int
    base_pos: np.ndarray
    pocket_center: np.ndarray
    pocket_radius: float
    Q: float
    d_flp: float
    theta_deg: float


# ==========================================================
# CORE FUNCTIONS (unchanged logic)
# ==========================================================

def read_structure(path):
    atoms = ase_read(path)
    return atoms

def read_nt2_nodes(path):
    nodes=[]
    with open(path) as f:
        for ln in f:
            ln=ln.strip()
            if not ln or ln.startswith("Vertex") or "#" in ln or "->" in ln or "table" in ln:
                continue
            try:
                p=ln.split()
                idx=int(p[0]); x=float(p[1]); y=float(p[2]); z=float(p[3]); r=float(p[4])
                if r>0:
                    nodes.append(VoronoiNode(idx,np.array([x,y,z]),r))
            except:
                continue
    return nodes

def detect_base(atoms):
    return [i for i,s in enumerate(atoms.get_chemical_symbols()) if s=="N"]


def distance_score(d,dmin,dmax):
    if d < dmin: return 0
    if d <= dmax: return 1
    return np.exp(-(d-dmax))

def lone_pair_direction(atoms,i,cut=2.2):
    pos = atoms.get_positions()
    b = pos[i]
    vec = pos - b
    dist = np.linalg.norm(vec,axis=1)
    neigh = vec[(dist>0.6)&(dist<cut)]
    if len(neigh)==0:
        return np.array([0,0,1])
    bonds = neigh/np.linalg.norm(neigh,axis=1)[:,None]
    v = -bonds.sum(axis=0)
    return v/np.linalg.norm(v)

def orient_score(lp,base,pocket,thetamax):
    v=pocket-base
    L=np.linalg.norm(v)
    if L<1e-8: return 0,0
    v/=L
    cos=np.clip(np.dot(lp,v),-1,1)
    theta=np.degrees(np.arccos(cos))
    if theta>thetamax: return 0,theta
    return cos*cos,theta

def accessibility(p,nodes,lam):
    if not nodes: return 0
    arr=np.array([nd.pos for nd in nodes])
    d=np.min(np.linalg.norm(arr-p,axis=1))
    return np.exp(-d/lam)


# ==========================================================
# FLP COMPUTATION (unchanged)
# ==========================================================

def compute_flp(atoms,nodes,r_acid):
    N_idx=detect_base(atoms)

    req = r_acid + DEFAULT_CLEARANCE
    acid_nodes=[n for n in nodes if n.radius>=req]
    co2_nodes =[n for n in nodes if n.radius>=DEFAULT_R_CO2]
    h2_nodes  =[n for n in nodes if n.radius>=DEFAULT_R_H2]

    if not acid_nodes:
        return []

    pos=atoms.get_positions()
    sites=[]

    for i in N_idx:
        base = pos[i]
        arr  = np.array([n.pos for n in acid_nodes])
        d    = np.linalg.norm(arr-base,axis=1)
        j    = np.argmin(d)
        node = acid_nodes[j]

        d_flp = np.linalg.norm(node.pos-base)-r_acid
        fd = distance_score(d_flp,DEFAULT_D_MIN,DEFAULT_D_MAX)
        if fd==0: continue

        lp = lone_pair_direction(atoms,i)
        fo,theta = orient_score(lp,base,node.pos,DEFAULT_THETA_MAX)
        if fo==0: continue

        fco2=accessibility(node.pos,co2_nodes,DEFAULT_LAMBDA_CO2)
        fh2 =accessibility(node.pos,h2_nodes ,DEFAULT_LAMBDA_H2)
        if fco2==0 or fh2==0: continue

        Q = fd*fo*fco2*fh2

        sites.append(FLPSite(
            base_index=i,
            base_pos=base,
            pocket_center=node.pos,
            pocket_radius=r_acid+DEFAULT_POCKET_BUF,
            Q=Q,
            d_flp=d_flp,
            theta_deg=theta
        ))

    return sites


def simultaneous(sites):
    sel=[]
    for s in sorted(sites,key=lambda x:-x.Q):
        clash=False
        for t in sel:
            if np.linalg.norm(s.pocket_center - t.pocket_center) < (s.pocket_radius+t.pocket_radius):
                clash=True; break
        if not clash:
            sel.append(s)
    return sel


# ==========================================================
# RESUME CHECK
# ==========================================================

def outputs_exist(cof_dir, cof, radius_tag):
    """Check if all three output files exist for a given radius."""
    summary = os.path.join(cof_dir,f"{cof}_FLP_{radius_tag}_summary.txt")
    table   = os.path.join(cof_dir,f"{cof}_FLP_{radius_tag}_sites.csv")
    jsonout = os.path.join(cof_dir,f"{cof}_FLP_visual_pockets_{radius_tag}.json")

    return os.path.exists(summary) and os.path.exists(table) and os.path.exists(jsonout)


# ==========================================================
# OUTPUT WRITER (unchanged except JSON format)
# ==========================================================

def save_outputs(cof_dir,cof,radius_tag,sites,sim_sites):

    summary = os.path.join(cof_dir,f"{cof}_FLP_{radius_tag}_summary.txt")
    table   = os.path.join(cof_dir,f"{cof}_FLP_{radius_tag}_sites.csv")
    json_out= os.path.join(cof_dir,f"{cof}_FLP_visual_pockets_{radius_tag}.json")

    C=sum(s.Q for s in sites)
    Cs=sum(s.Q for s in sim_sites)
    with open(summary,"w") as f:
        f.write(f"COF : {cof}\nRadius label : {radius_tag}\n\n")
        f.write(f"N_base_FLP      = {len(sites)}\n")
        f.write(f"C_FLP           = {C:.6f}\n")
        f.write(f"N_base_FLP_sim  = {len(sim_sites)}\n")
        f.write(f"C_FLP_sim       = {Cs:.6f}\n")

    with open(table,"w") as f:
        f.write("N,d_flp,theta,Q,category,x,y,z\n")
        for s in sites:
            f.write(f"{s.base_index+1},{s.d_flp:.3f},{s.theta_deg:.3f},{s.Q:.5f},"
                    f"x={s.pocket_center[0]:.3f},{s.pocket_center[1]:.3f},{s.pocket_center[2]:.3f}\n")

    # JSON: same format as single-cof
    data={
        "simultaneous_flp_sites":[
            {
                "N_index": s.base_index+1,
                "x": float(s.pocket_center[0]),
                "y": float(s.pocket_center[1]),
                "z": float(s.pocket_center[2]),
                "radius": float(s.pocket_radius),
                "Q": float(s.Q)
            }
            for s in sim_sites
        ]
    }
    with open(json_out,"w") as f:
        json.dump(data,f,indent=2)


# ==========================================================
# MAIN PER-COF PROCESSOR (FULLY RESUMABLE)
# ==========================================================

def run_one(folder):

    try:
        cif = [f for f in os.listdir(folder) if f.endswith(".cif")][0]
        cof = cif.replace(".cif","")

        cif_path=os.path.join(folder,cif)
        nt2_path=os.path.join(folder,cof+".nt2")

        # Run network only if nt2 missing
        if not os.path.exists(nt2_path):
            subprocess.run(["network","-nt2",cif],cwd=folder,check=True)

        atoms=read_structure(cif_path)
        nodes=read_nt2_nodes(nt2_path)

        # Loop through radii
        for tag,r in ACID_SIZES.items():

            if outputs_exist(folder,cof,tag):
                print(f"[SKIP] {cof} / {tag} already complete.")
                continue

            print(f"[RUN] {cof} / {tag} missing outputs → computing...")
            sites = compute_flp(atoms,nodes,r)
            sims  = simultaneous(sites)
            save_outputs(folder,cof,tag,sites,sims)

        return cof,"OK"

    except Exception as e:
        return os.path.basename(folder),f"FAILED — {e}"


# ==========================================================
# BATCH EXECUTION
# ==========================================================

def main():

    folders=[os.path.join(ROOT_DIR,d)
             for d in os.listdir(ROOT_DIR)
             if os.path.isdir(os.path.join(ROOT_DIR,d))
             and any(f.endswith(".cif") for f in os.listdir(os.path.join(ROOT_DIR,d)))]

    print(f"\nFOUND {len(folders)} COFs → Starting Fully Resumable FLP Processing\n")

    results=[]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs={exe.submit(run_one,f):f for f in folders}
        if TQDM:
            for f in tqdm(as_completed(futs),total=len(futs)):
                results.append(f.result())
        else:
            for f in as_completed(futs):
                results.append(f.result())

    print("\n=========== FINAL SUMMARY ===========")
    for name,status in results:
        print(f"{name:<15} → {status}")


if __name__=="__main__":
    main()
