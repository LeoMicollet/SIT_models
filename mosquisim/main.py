# models.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


from data import time_data, precip_data, hum_data, max_rain
from bio import allee, competition1
from models import Ms_fun, sim_7, sim_2
import params

# ── Simulation setup ──────────────────────────────────────────────
days = np.arange(0, 720)

# Initial conditions
init_8 = [10000, 20000, 15000, 500, 0, 17000, 5000]   # E,L,P,F,Ff,Fs,M
init_3 = [17500, 5000]                        # F, M

# Release schedule
#release_times = None
release_times = np.arange(400, 365+140, 7)
rho = 10000

p = params.params   # shorthand

# ── Run both models ───────────────────────────────────────────────
sol8 = sim_7(
    pop_init=init_8, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_pa=p["transi_pa"],
    death_L=p["death_L"], death_P=p["death_P"],
    c=p["c"], mu=p["mu"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data
)

sol3 = sim_2(
    pop_init=init_3, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_mod=p["transi_mod"],
    death_L=p["death_L"],
    c=p["c"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data
)

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Full model
ax = axes[0]
ax.plot(days, sol8[3]+ sol8[4] + sol8[5],  label="F  (wild females)")
ax.plot(days, sol8[6],  label="M  (wild males)")
ax.plot(days, sol8[7],  label="Ms (sterile males)")
ax.set_title("Full model (7 dim)")
ax.set_ylabel("Population")
ax.legend()

# Reduced model
ax = axes[1]
ax.plot(days, sol3[0],  label="F  (wild females)")
ax.plot(days, sol3[1],  label="M  (wild males)")
ax.plot(days, sol3[2],  label="Ms (sterile males)")
ax.set_title("Reduced model (2 dim)")
ax.set_ylabel("Population")
ax.legend()

# Differences (full model − reduced model)
F8   = sol8[3] + sol8[4] + sol8[5]   # total females in 7-dim model
M8   = sol8[6]
F3   = sol3[0]
M3   = sol3[1]

ax = axes[2]
ax.plot(days, F8 - F3, label="ΔF  (females: full − reduced)")
ax.plot(days, M8 - M3, label="ΔM  (males:   full − reduced)")
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_title("Difference between models (full − reduced)")
ax.set_ylabel("Population difference")
ax.set_xlabel("Day")
ax.legend()

plt.tight_layout()
plt.savefig("simu.png", dpi=150)   # saves a file — useful outside notebooks
plt.show()