import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# -------------------------
# Import your training modules (must be present in same folder)
# -------------------------
from dataset import COFModel1Dataset
from model import COFModel1

from dataset_model1b import COFModel1bDataset
from model_model1b import COFModel1b

from dataset_model2_v2 import COFModel2v2Dataset

# IMPORTANT: you must have the model2 v2 definition file used for training
# Typical name:
from model_model2_v2 import COFModel2v2  # <- keep the same class you trained with


FEATURES = {
    "2c_fn": "features/2_con_fn_linker_feat_ms.csv",
    "2c_unfn": "features/2_con_unfn_linker_feat_ms.csv",
    "3c": "features/3_con_linker_feat_ms.csv",
    "4c": "features/4_con_linker_feat_ms.csv",
    "base": "features/base_feat_ms.csv",
}


def _ensure_targets_for_dataset(df: pd.DataFrame):
    # Model-1 needs LCD & PLD columns (even for inference)
    if "LCD" not in df.columns:
        df["LCD"] = 0.0
    if "PLD" not in df.columns:
        df["PLD"] = 0.0

    # Model-1b needs L_AV_frac column (even for inference)
    if "L_AV_frac" not in df.columns:
        df["L_AV_frac"] = 0.0

    # Model-2 v2 default target col: C_FLP_sim (even for inference)
    if "C_FLP_sim" not in df.columns:
        df["C_FLP_sim"] = 0.0

    return df


def _load_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {path} is not in expected format (dict with model_state_dict).")
    return ckpt


@torch.no_grad()
def predict_model1(meta_csv, embed_dir, device="cpu", batch_size=256):
    ckpt = _load_ckpt("model1_final.pt", device)
    ds = COFModel1Dataset(meta_csv, FEATURES, embed_dir, norm_stats=ckpt.get("norm_stats", None))
    dims = ckpt.get("dims", None)
    if dims is None:
        sample = ds[0]
        dims = {k: int(sample[k].shape[0]) for k in ["topo", "node", "linker", "base", "misc"]}

    model = COFModel1(dims=dims, hidden=128).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    for batch in dl:
        for k in ["topo", "node", "linker", "base", "misc"]:
            batch[k] = batch[k].to(device)
        yhat = model(batch).detach().cpu().numpy()
        preds.append(yhat)
    preds = np.vstack(preds)  # (N, 2) -> [LCD, PLD]
    return preds[:, 0], preds[:, 1]


@torch.no_grad()
def predict_model1b(meta_csv, embed_dir, device="cpu", batch_size=256):
    ckpt = _load_ckpt("model1b_L_avfrac_final.pt", device)
    ds = COFModel1bDataset(meta_csv, FEATURES, embed_dir, norm_stats=ckpt.get("norm_stats", None))
    dims = ckpt.get("dims", None)
    if dims is None:
        sample = ds[0]
        dims = {k: int(sample[k].shape[0]) for k in ["topo", "node", "linker", "base", "misc"]}

    model = COFModel1b(dims=dims, hidden=128).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    for batch in dl:
        for k in ["topo", "node", "linker", "base", "misc"]:
            batch[k] = batch[k].to(device)
        yhat = model(batch).detach().cpu().numpy().reshape(-1)
        preds.append(yhat)
    return np.concatenate(preds)


@torch.no_grad()
def predict_model2v2(meta_csv, embed_dir, device="cpu", batch_size=256):
    ckpt = _load_ckpt("model2_v2_flp_large_final.pt", device)

    # If your ckpt stores target_mode/target_col, respect it; otherwise defaults are used
    target_col = ckpt.get("target_col", "C_FLP_sim")
    target_mode = ckpt.get("target_mode", "log1p")

    ds = COFModel2v2Dataset(
        meta_csv,
        FEATURES,
        embed_dir,
        norm_stats=ckpt.get("norm_stats", None),
        target_col=target_col,
        target_mode=target_mode
    )
    dims = ckpt.get("dims", None)
    if dims is None:
        sample = ds[0]
        dims = {k: int(sample[k].shape[0]) for k in ["topo", "node", "linker", "base", "misc"]}

    # Must match the class used in training
    model = COFModel2v2(dims=dims, hidden=ckpt.get("hidden", 128)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    for batch in dl:
        for k in ["topo", "node", "linker", "base", "misc"]:
            batch[k] = batch[k].to(device)
        yhat = model(batch).detach().cpu().numpy().reshape(-1)
        preds.append(yhat)
    yhat = np.concatenate(preds)

    # undo target transform if log1p was used
    if target_mode == "log1p":
        yhat = np.expm1(yhat)

    return yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop_csv", required=True)
    ap.add_argument("--embed_dir", default="embed")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_csv(args.pop_csv)
    df = _ensure_targets_for_dataset(df)

    tmp_meta = "_tmp_ga_meta_for_prediction.csv"
    df.to_csv(tmp_meta, index=False)

    lcd_pred, pld_pred = predict_model1(tmp_meta, args.embed_dir, device=args.device, batch_size=args.batch_size)
    lav_pred = predict_model1b(tmp_meta, args.embed_dir, device=args.device, batch_size=args.batch_size)
    flp_pred = predict_model2v2(tmp_meta, args.embed_dir, device=args.device, batch_size=args.batch_size)

    os.remove(tmp_meta)

    out = df.copy()
    out["LCD_pred"] = lcd_pred
    out["PLD_pred"] = pld_pred
    out["L_AV_frac_pred"] = lav_pred
    out["C_FLP_sim_pred"] = flp_pred

    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved predictions: {args.out_csv} (N={len(out)})")


if __name__ == "__main__":
    main()
