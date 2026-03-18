# models.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


from data import time_data, precip_data, hum_data, max_rain
from bio import allee, competition1
from models import Ms_fun, sim_7, sim_2, pre_release_adult_input
import params

# ── Simulation setup ──────────────────────────────────────────────
days = np.arange(0, 720)

# Initial conditions
init_8 = [10000, 20000, 15000, 500, 0, 17000, 5000]   # E,L,P,F,Ff,Fs,M
init_3 = [17500, 5000]                        # F, M

# Release schedule
#release_times = None
release_times = [400]
#release_times = np.arange(400, 365+140, 7)
rho = 100000

p = params.params   # shorthand

# ── Run both models ───────────────────────────────────────────────
sol8 = sim_7(
    pop_init=init_8, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_pa=p["transi_pa"],
    death_L=p["death_L"], death_P=p["death_P"],
    c=p["c"], mu=p["mu"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data, 
    type = 3
)

# New reduced model: bootstrap from 7D state at first release
sol3_new = sim_2(
    pop_init=init_3, days=days,
    birth=p["birth"], deltaA=p["deltaA"], deltaE=p["deltaE"],
    transi_el=p["transi_el"], transi_lp=p["transi_lp"], transi_mod=p["transi_mod"],
    death_L=p["death_L"],
    c=p["c"], n_egg=p["n_egg"],
    release_times=release_times, rho=rho,
    precip_data=precip_data, H=hum_data,
    pre_release_pop_init_7d=init_8,
    transi_pa=p["transi_pa"], death_P=p["death_P"], mu=p["mu"],
    type = 3
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
    type = 3
)

# ── Plot ──────────────────────────────────────────────────────────
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

# Differences (full model − reduced model)
F8   = sol8[3] + sol8[4] + sol8[5]   # total females in 7-dim model
M8   = sol8[6]
F3   = sol3_new[0]
M3   = sol3_new[1]

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


# ── Old vs New reduced model comparison ──────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax = axes[0]
ax.plot(days, sol3_old[0], label="F old", color="tab:blue", alpha=0.8)
ax.plot(days, sol3_new[0], label="F new", color="tab:blue", linestyle="--", linewidth=1.8)
ax.plot(days, sol3_old[1], label="M old", color="tab:red", alpha=0.8)
ax.plot(days, sol3_new[1], label="M new", color="tab:red", linestyle="--", linewidth=1.8)
ax.axvline(release_times[0], color="k", linestyle=":", linewidth=1.0, label="first release")
ax.set_title("Old vs New reduced model")
ax.set_ylabel("Population")
ax.legend(ncol=3)

ax = axes[1]
ax.plot(days, sol3_new[0] - sol3_old[0], label="ΔF (new-old)", color="tab:blue")
ax.plot(days, sol3_new[1] - sol3_old[1], label="ΔM (new-old)", color="tab:red")
ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
ax.axvline(release_times[0], color="k", linestyle=":", linewidth=1.0)
ax.set_title("Difference between new and old reduced model")
ax.set_xlabel("Day")
ax.set_ylabel("Population difference")
ax.legend()

plt.tight_layout()
plt.savefig("old_new_compare.png", dpi=150)
plt.show()


# Zoom around first release for easier visual validation
zoom_left = max(days[0], release_times[0] - 30)
zoom_right = min(days[-1], release_times[0] + 60)
mask = (days >= zoom_left) & (days <= zoom_right)

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax.plot(days[mask], sol3_old[0][mask], label="F old", color="tab:blue", alpha=0.8)
ax.plot(days[mask], sol3_new[0][mask], label="F new", color="tab:blue", linestyle="--", linewidth=1.8)
ax.plot(days[mask], sol3_old[1][mask], label="M old", color="tab:red", alpha=0.8)
ax.plot(days[mask], sol3_new[1][mask], label="M new", color="tab:red", linestyle="--", linewidth=1.8)
ax.axvline(release_times[0], color="k", linestyle=":", linewidth=1.0, label="first release")
ax.set_title("Old vs New (zoom around first release)")
ax.set_xlabel("Day")
ax.set_ylabel("Population")
ax.legend(ncol=3)

plt.tight_layout()
plt.savefig("old_new_zoom.png", dpi=150)
plt.show()

# ── Pre-release adult input ───────────────────────────────────────
first_release = float(release_times[0])
idx_rel = int(np.searchsorted(days, first_release))
E0, L0, P0, F0, Ff0, Fs0, M0 = sol8[:7, idx_rel]

params_residual = {
    'delta_F': p["deltaA"],
    'tau_E':   p["transi_el"],
    'delta_E': p["deltaE"],
    'tau_L':   p["transi_lp"],
    'delta_L': p["death_L"],
    'tau_P':   p["transi_pa"],
    'delta_P': p["death_P"],
    'beta':    p["n_egg"] * p["birth"],
    'mu':      p["mu"],
}
initial_state_residual = [Ff0, E0, L0, P0]

post_days = days[days >= first_release]
f_flux = np.array([pre_release_adult_input(t, first_release, initial_state_residual, params_residual)[0] for t in post_days])
m_flux = np.array([pre_release_adult_input(t, first_release, initial_state_residual, params_residual)[1] for t in post_days])

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax.plot(post_days, f_flux , label="female flux", color="tab:blue")
ax.plot(post_days, m_flux , label="male flux", color="tab:red")
ax.axvline(first_release, color="k", linestyle=":", linewidth=1.0, label="first release")
ax.set_title("Pre-release adult input (residual cohort emergence)")
ax.set_xlabel("Day")
ax.set_ylabel("Adult flux (individuals/day)")
ax.legend()
plt.tight_layout()
plt.savefig("pre_release_input.png", dpi=150)
plt.show()