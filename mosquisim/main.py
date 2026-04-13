# models.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


from data import time_data, precip_data, hum_data, max_rain
from bio import allee, competition1
from models import Ms_fun, sim_7, sim_2, pre_release_adult_input
import params

#  Simulation setup 
days = np.arange(0, 365*4)

# Initial conditions
init_8 = [182600, 11900, 4200, 700, 15000, 0, 5000]   # E,L,P,F,Ff,Fs,M
init_3 = [16000, 5300]                        # F, M

# Release schedule
#release_times = None
release_times = [200]
#release_times = np.arange(400, 365+140, 7)
rho = 50000

p = params.params   # shorthand
delay = 1/(p["transi_pa"]+ p["death_P"]) + 1/(p["transi_lp"] + p["death_L"]) + 1/(p["transi_el"] + p["deltaE"])
delay = 4*( 1/(p["transi_pa"]) + 1/(p["transi_lp"]) * (init_8[0] + init_8[1])/(init_8[0] + init_8[1] + init_8[2]) + 1/(p["transi_el"])* (init_8[0])/(init_8[0] + init_8[1] + init_8[2]))
delay = 1/(p["transi_pa"]) + 1/(p["transi_lp"] ) + 1/(p["transi_el"])

if release_times is not None:
    release_delay = release_times[0] + delay

else :
    release_delay = None
#  Run both models 
sol8 = sim_7(
    pop_init=init_8, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_pa=p["transi_pa"],
    death_L=p["death_L"], death_P=p["death_P"],
    c=p["c"], mu=p["mu"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data, 
    type = 3, Sterile = 0
)

# New reduced model: bootstrap from 7D state at first release
sol3_new = sim_2(
    pop_init=init_3, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_mod=p["transi_mod"],
    death_L=p["death_L"],
    c=p["c"], n_egg=p["n_egg"],
    release_times=release_delay, rho=rho,
    precip_data=precip_data, H=hum_data,
    reltype = 3, Sterile = 0
)

# Old reduced model: no 7D bootstrap at first release
sol3_old = sim_2(
    pop_init=init_3, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_mod=p["transi_mod"],
    death_L=p["death_L"],
    c=p["c"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data,
    reltype = 3, Sterile = 0
)

print(sol8[:, -1])
print(sol3_new[:, -1])
#  Plot 
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Full model
ax = axes[0]
ax_ms = ax.twinx()
line_f = ax.plot(days, sol8[3] + sol8[4] + sol8[5], label="F  (wild females)", color="tab:blue")
line_m = ax.plot(days, sol8[6], label="M  (wild males)", color="tab:red")
line_ms = ax_ms.plot(days, sol8[7], label="Ms (sterile males)", color="tab:orange", linestyle="--")
ax.set_title("Full model (7 dim)")
ax.set_ylabel("F, M population")
ax_ms.set_ylabel("Ms population")
lines = line_f + line_m + line_ms
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper right")

# Reduced model
ax = axes[1]
ax_ms = ax.twinx()
line_f = ax.plot(days, sol3_new[0], label="F  (wild females)", color="tab:blue")
line_m = ax.plot(days, sol3_new[1], label="M  (wild males)", color="tab:red")
line_ms = ax_ms.plot(days, sol3_new[2], label="Ms (sterile males)", color="tab:orange", linestyle="--")
ax.set_title("Reduced model (2 dim)")
ax.set_ylabel("F, M population")
ax_ms.set_ylabel("Ms population")
lines = line_f + line_m + line_ms
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper right")

# Differences (full model - reduced model)
F8   = sol8[3] + sol8[4] + sol8[5]   # total females in 7-dim model
M8   = sol8[6]
F3   = sol3_new[0]
M3   = sol3_new[1]

ax = axes[2]
ax.plot(days, F8 - F3, label="ΔF  (females: full - reduced)")
ax.plot(days, M8 - M3, label="ΔM  (males:   full - reduced)")
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.set_title("Difference between models (full - reduced)")
ax.set_ylabel("Population difference")
ax.set_xlabel("Day")
ax.legend()

plt.tight_layout()
plt.savefig("simu.png", dpi=150)   # saves a file - useful outside notebooks
plt.show()

# Zoom around first release for easier visual validation
zoom_left = max(days[0], release_times[0] - 30)
zoom_right = min(days[-1], release_times[0] + 500)
mask = (days >= zoom_left) & (days <= zoom_right)
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

# Females plot
ax = axes[0]
ax.plot(days[mask], sol8[3][mask] + sol8[4][mask] + sol8[5][mask], label="F full model", color="tab:blue", alpha=0.8)
ax.plot(days[mask], sol3_new[0][mask], label="F reduced model", color="tab:blue", linestyle="--", linewidth=1.8)
ax.axvline(release_times[0], color="k", linestyle=":", linewidth=1.0, label="first release")
ax.set_title("Females (zoom around first release)")
ax.set_ylabel("Population")
ax.legend()

# Males plot
ax = axes[1]
ax.plot(days[mask], sol8[6][mask], label="M full model", color="tab:red", alpha=0.8)
ax.plot(days[mask], sol3_new[1][mask], label="M reduced model", color="tab:red", linestyle="--", linewidth=1.8)
ax.axvline(release_times[0], color="k", linestyle=":", linewidth=1.0, label="first release")
ax.set_title("Males (zoom around first release)")
ax.set_xlabel("Day")
ax.set_ylabel("Population")
ax.legend()

plt.tight_layout()
plt.savefig("old_new_zoom.png", dpi=150)
plt.show()
