import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

from torch.serialization import add_safe_globals
from torch_geometric.data import Data

add_safe_globals([Data])


class COFModel1Dataset(Dataset):
    """
    COFModel1Dataset with optional feature-family masking inside node/linker/base vectors.

    feature_family_mode:
      - "all" (default): use all features
      - "rdf_only": keep RDF features only
      - "geom_only": keep geom_* features only
      - "mordred_only": keep all non-RDF and non-geom features
      - "no_rdf": remove RDF features
      - "no_geom": remove geom_* features
      - "no_mordred": remove all non-RDF and non-geom features
    """

    def __init__(self, meta_csv, feature_paths, embed_dir, norm_stats=None, feature_family_mode="all"):
        self.meta = pd.read_csv(meta_csv)
        self.embed_dir = Path(embed_dir)
        self.norm_stats = norm_stats

        # Feature tables
        self.fn2c = pd.read_csv(feature_paths['2c_fn']).set_index('linker_id')
        self.unfn2c = pd.read_csv(feature_paths['2c_unfn']).set_index('linker_id')
        self.l3c = pd.read_csv(feature_paths['3c']).set_index('linker_id')
        self.l4c = pd.read_csv(feature_paths['4c']).set_index('linker_id')
        self.base = pd.read_csv(feature_paths['base']).set_index('linker_id')

        # Drop non-numeric label column if present
        self.fn2c_cols = self.fn2c.columns.drop('name_x', errors='ignore')
        self.unfn2c_cols = self.unfn2c.columns.drop('name_x', errors='ignore')
        self.l3c_cols = self.l3c.columns.drop('name_x', errors='ignore')
        self.l4c_cols = self.l4c.columns.drop('name_x', errors='ignore')
        self.base_cols = self.base.columns.drop('name_x', errors='ignore')

        # Misc (functionalization metadata)
        self.bridge_types = ['none', 'ch2', 'ph', 'dir']
        self.bridge_map = {b: i for i, b in enumerate(self.bridge_types)}

        # Feature family mode
        self.feature_family_mode = feature_family_mode

        # Precompute index sets for feature families for each table
        # (We assume RDF_... and geom_... prefixes as observed in your CSVs)
        self._family_idx = {
            "node3": self._compute_family_indices(self.l3c_cols),
            "node4": self._compute_family_indices(self.l4c_cols),
            "parent2c": self._compute_family_indices(self.unfn2c_cols),
            "edge2c": self._compute_family_indices(self.fn2c_cols),
            "base": self._compute_family_indices(self.base_cols),
        }

        # For linker_vec = [parent_2c, edge_fn], we need combined indices with offset
        self._family_idx["linker_combined"] = self._combine_linker_family_indices(
            self._family_idx["parent2c"], len(self.unfn2c_cols),
            self._family_idx["edge2c"], len(self.fn2c_cols)
        )

    def __len__(self):
        return len(self.meta)

    @staticmethod
    def _compute_family_indices(cols):
        cols = list(cols)
        geom_idx = [i for i, c in enumerate(cols) if str(c).startswith("geom_")]
        rdf_idx = [i for i, c in enumerate(cols) if str(c).startswith("RDF_")]
        # Mordred = everything else (numeric descriptors not in RDF_ or geom_)
        mord_idx = [i for i in range(len(cols)) if (i not in geom_idx) and (i not in rdf_idx)]
        return {"geom": np.array(geom_idx, dtype=int),
                "rdf": np.array(rdf_idx, dtype=int),
                "mordred": np.array(mord_idx, dtype=int),
                "n": len(cols)}

    @staticmethod
    def _combine_linker_family_indices(parent_fam, n_parent, edge_fam, n_edge):
        # parent indices in [0..n_parent-1]
        # edge indices shifted by +n_parent
        geom = np.concatenate([parent_fam["geom"], edge_fam["geom"] + n_parent]).astype(int)
        rdf = np.concatenate([parent_fam["rdf"], edge_fam["rdf"] + n_parent]).astype(int)
        mord = np.concatenate([parent_fam["mordred"], edge_fam["mordred"] + n_parent]).astype(int)
        return {"geom": geom, "rdf": rdf, "mordred": mord, "n": n_parent + n_edge}

    def _get_vec(self, df, idx, cols):
        if idx == -1:
            return torch.zeros(len(cols), dtype=torch.float32)
        series = df.loc[idx, cols]
        numeric = pd.to_numeric(series, errors='coerce').fillna(0.0)
        return torch.tensor(numeric.values, dtype=torch.float32)

    def _norm(self, x, key):
        if self.norm_stats is None:
            return x
        return (x - self.norm_stats[key]['mean']) / self.norm_stats[key]['std']

    def _apply_family_mode(self, x, fam_idx):
        """
        Apply feature_family_mode to vector x using indices fam_idx (geom/rdf/mordred).
        Masking is done by zeroing selected columns.
        """
        mode = self.feature_family_mode
        if (mode is None) or (mode == "all"):
            return x

        x = x.clone()

        geom = fam_idx["geom"]
        rdf = fam_idx["rdf"]
        mord = fam_idx["mordred"]

        if mode == "rdf_only":
            keep = set(rdf.tolist())
            for idx in range(x.shape[0]):
                if idx not in keep:
                    x[idx] = 0.0

        elif mode == "geom_only":
            keep = set(geom.tolist())
            for idx in range(x.shape[0]):
                if idx not in keep:
                    x[idx] = 0.0

        elif mode == "mordred_only":
            keep = set(mord.tolist())
            for idx in range(x.shape[0]):
                if idx not in keep:
                    x[idx] = 0.0

        elif mode == "no_rdf":
            if len(rdf) > 0:
                x[rdf] = 0.0

        elif mode == "no_geom":
            if len(geom) > 0:
                x[geom] = 0.0

        elif mode == "no_mordred":
            if len(mord) > 0:
                x[mord] = 0.0

        else:
            raise ValueError(f"Unknown feature_family_mode: {mode}")

        return x

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # ---- topology embedding (NO normalization)
        topo_obj = torch.load(row['topology_pt_path'], weights_only=False)
        topo_vec = topo_obj.x.mean(dim=0).float() if hasattr(topo_obj, 'x') else topo_obj.float()

        # ---- nodes
        if row['node1_connectivity'] == 3:
            n1 = self._get_vec(self.l3c, row['node1_linker_id'], self.l3c_cols)
            fam_n1 = self._family_idx["node3"]
        else:
            n1 = self._get_vec(self.l4c, row['node1_linker_id'], self.l4c_cols)
            fam_n1 = self._family_idx["node4"]

        if row['node2_connectivity'] == 3:
            n2 = self._get_vec(self.l3c, row['node2_linker_id'], self.l3c_cols)
            fam_n2 = self._family_idx["node3"]
        else:
            n2 = self._get_vec(self.l4c, row['node2_linker_id'], self.l4c_cols)
            fam_n2 = self._family_idx["node4"]

        node_vec = 0.5 * (n1 + n2)
        node_vec = self._norm(node_vec, 'node')
        # Apply feature-family masking to node vector
        # (If node1/node2 had different connectivity, we still mask by the vector length; both share same 812 anyway.)
        node_vec = self._apply_family_mode(node_vec, fam_n1)

        # ---- linkers
        parent_2c = self._get_vec(self.unfn2c, row['parent_2c_id'], self.unfn2c_cols)
        edge_fn = self._get_vec(self.fn2c, row['edge_fn_id'], self.fn2c_cols)
        linker_vec = torch.cat([parent_2c, edge_fn])
        linker_vec = self._norm(linker_vec, 'linker')
        # Apply feature-family masking to combined linker vector
        linker_vec = self._apply_family_mode(linker_vec, self._family_idx["linker_combined"])

        # ---- base
        base_vec = self._get_vec(self.base, row['base_id'], self.base_cols)
        base_vec = self._norm(base_vec, 'base')
        base_vec = self._apply_family_mode(base_vec, self._family_idx["base"])

        # ---- misc (NO normalization)
        bridge = torch.zeros(len(self.bridge_types))
        bridge[self.bridge_map[row['bridge_type']]] = 1.0
        misc_vec = torch.cat([
            torch.tensor([row['coverage_fraction']], dtype=torch.float32),
            bridge
        ])

        y = torch.tensor([row['LCD'], row['PLD']], dtype=torch.float32)

        return {
            'topo': topo_vec,
            'node': node_vec,
            'linker': linker_vec,
            'base': base_vec,
            'misc': misc_vec,
            'y': y
        }

