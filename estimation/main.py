# main.py
from estimation import (estimate_simple, simulate_and_estimate_simple,
                         estimate_full,  simulate_and_estimate_full)
import numpy as np
import pandas as pd

# ── Setting 1: simple ─────────────────────────────────────────────
# On real data
file_rel = '/home/leo/Documents/These/Codes/SIT_models/Data/df_poly_release.csv' 
file_capt = '/home/leo/Documents/These/Codes/SIT_models/Data/df_poly_capture.csv'
df_rel = pd.read_csv(file_rel, sep=',')
df_capt = pd.read_csv(file_capt, sep=',')

val = estimate_simple(df_rel, df_capt["Nb_ind"].values, df_capt["Week"].values)
print(f"alpha={val.x[0]:.5f}, delta_M={val.x[1]:.5f}")

# Monte Carlo validation
results_df = simulate_and_estimate_simple(
    df_rel, delta_true=0.1, alpha_true=0.0001,
    weeks=np.arange(1, 53), n_simulations=50
)

# ── Setting 2: full ───────────────────────────────────────────────
week_starts = np.arange(0, 53*7, 7)   # day 0, 7, 14, ..., 364
sim_days    = np.arange(0, 365, 1.0)

est = estimate_full(
    F0=15000, M0=5000, nu=0.5,
    release_times_days=[200, 207, 214],
    rho=50000,
    captures_M=capt_M_obs,   # your observed weekly male captures
    captures_F=capt_F_obs,
    week_starts_days=week_starts,
    sim_days=sim_days,
    A0=1.0, B0=1.0, C0=1.0,
    delta_M0=0.1, delta_F0=0.1,
    alpha_M0=1e-4, alpha_F0=1e-4
)
print(est)