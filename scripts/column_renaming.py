import pandas as pd

# ==========================
# INPUT / OUTPUT
# ==========================
INFILE  = "global_props_opt.csv"
OUTFILE = "global_props_opt_clean.csv"

df = pd.read_csv(INFILE)

rename_map = {}

for col in df.columns:
    new = col

    # ---------- Channel ----------
    new = new.replace("_channel_dim", "_CHAN_dim")
    new = new.replace("_probe_rad", "_CHAN_probe_rad_A")
    new = new.replace("_probe_diam", "_CHAN_probe_diam_A")

    # ---------- Surface area ----------
    new = new.replace("_SA_Unitcell_volume", "_SA_cell_volume_A3")
    new = new.replace("_SA_Density", "_SA_density_g_cm3")
    new = new.replace("_SA_2_per_g", "_SA_ASA_m2_g")
    new = new.replace("_SA_3", "_SA_ASA_m2_cm3")
    new = new.replace("_SA_2", "_SA_ASA_A2")
    new = new.replace("_SA_Number_of_channels", "_SA_n_channels")
    new = new.replace("_SA_Number_of_pockets", "_SA_n_pockets")

    # ---------- Accessible volume ----------
    new = new.replace("_VOL_Unitcell_volume", "_AV_cell_volume_A3")
    new = new.replace("_VOL_Density", "_AV_density_g_cm3")
    new = new.replace("_VOL_3_per_g", "_AV_cm3_g")
    new = new.replace("_VOL_AV_Volume_fraction", "_AV_frac")
    new = new.replace("_VOL_3", "_AV_A3")
    new = new.replace("_VOL_NAV_Volume_fraction", "_NAV_frac")
    new = new.replace("_VOL_Number_of_channels", "_AV_n_channels")
    new = new.replace("_VOL_Number_of_pockets", "_AV_n_pockets")

    # ---------- Probe occupiable ----------
    new = new.replace("_VOLPO_Unitcell_volume", "_POAV_cell_volume_A3")
    new = new.replace("_VOLPO_Density", "_POAV_density_g_cm3")
    new = new.replace("_VOLPO_3_per_g", "_POAV_cm3_g")
    new = new.replace("_VOLPO_POAV_Volume_fraction", "_POAV_frac")
    new = new.replace("_VOLPO_3", "_POAV_A3")
    new = new.replace("_VOLPO_PONAV_Volume_fraction", "_PONAV_frac")

    # ---------- Probe statistics ----------
    new = new.replace("_VOLPO_PO_filename", "_PO_file")
    new = new.replace("_VOLPO_PO_density_g_cm3", "_PO_density_g_cm3")
    new = new.replace("_VOLPO_PO_probe_rad", "_PO_probe_rad_A")
    new = new.replace("_VOLPO_PO_n_points", "_PO_n_MC")
    new = new.replace("_VOLPO_PO_probe_ctr_A_fract", "_PO_ctr_A_frac")
    new = new.replace("_VOLPO_PO_probe_ctr_NA_fract", "_PO_ctr_NA_frac")
    new = new.replace("_VOLPO_PO_A_fract", "_PO_A_frac")
    new = new.replace("_VOLPO_PO_NA_fract", "_PO_NA_frac")
    new = new.replace("_VOLPO_PO_narrow_fract", "_PO_narrow_frac")
    new = new.replace("_VOLPO_PO_ovlp_fract", "_PO_overlap_frac")

    rename_map[col] = new

df = df.rename(columns=rename_map)
df.to_csv(OUTFILE, index=False)

print(f"✅ Cleaned file written → {OUTFILE}")
print(f"Columns: {len(df.columns)}")
