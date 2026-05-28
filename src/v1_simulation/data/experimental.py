# src/v1_simulation/data/experimental.py
from pathlib import Path

import numpy as np


class ExperimentalData:
    def __init__(self, cfg_model, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Experimental data file not found: {self.path}")

        self.data = np.load(self.path, allow_pickle=True)

        self.eta_I = self.data["eta_I"]
        self.eta_X = self.data["eta_X"]

        self.N_X = cfg_model.layers.l4.n_side**2
        exact_N_E = self.N_X / self.eta_X
        exact_N_I = exact_N_E * self.eta_I
        self.l2_3_n_side = int(np.ceil(np.sqrt(exact_N_E + exact_N_I)))

        n_total = self.l2_3_n_side**2
        self.N_E = int(n_total / (1 + self.eta_I))
        self.N_I = n_total - self.N_E

        self.p_EE = cfg_model.connectivity.p_ee

        self.NT_X = int(self.data["etaT_X"] * self.N_X)
        self.pT_X = self.NT_X / self.N_X # 0.5714