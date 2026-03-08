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
init_8 = [100, 50, 20, 80, 0, 0, 40]   # E,L,P,F,Ff,Fs,M
init_3 = [80, 40]                        # F, M

# Release schedule
release_times = np.arange(365, 365+210, 7)
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
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

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
ax.set_xlabel("Day")
ax.legend()

plt.tight_layout()
plt.savefig("simu.png", dpi=150)   # saves a file — useful outside notebooks
plt.show()