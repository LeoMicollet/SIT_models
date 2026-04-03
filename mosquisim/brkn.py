# models.py
import numpy as np
from scipy.integrate import solve_ivp

from data import time_data, precip_data, hum_data, max_rain
from bio import allee, competition1
# ─────────────────────────────────────────────────────────────────
# Ms analytical solution — no ODE, no solver
# ─────────────────────────────────────────────────────────────────

def Ms_fun(t, release_times, rho, deltaA):
    """
    M_s(t) = sum_{i: t_i <= t} rho_i * exp(-d_M * (t - t_i))
    
    Parameters
    ----------
    t             : float or np.ndarray – evaluation time(s)
    release_times : array-like           – t_i, times of releases
    rho           : float or array-like  – release size(s); scalar = constant
    deltaA        : float                – adult death rate d_M = 3 * 1.2 * deltaA

    Returns
    -------
    Ms : float or np.ndarray (same shape as t)
    """
    d_M = 3 * 1.2 * deltaA
    release_times = np.asarray(release_times)

    # Allow constant or per-release sizes
    if np.isscalar(rho):
        rho = np.full(len(release_times), rho)
    else:
        rho = np.asarray(rho)

    scalar_input = np.isscalar(t)
    t = np.atleast_1d(np.asarray(t, dtype=float))

    Ms = np.zeros(len(t))
    for i, ti in enumerate(release_times):
        mask = t >= ti
        Ms[mask] += rho[i] * np.exp(-d_M * (t[mask] - ti))

    return Ms[0] if scalar_input else Ms


def compute_P_exact(t_i, t_1, initial_state, params):
    """
    Exact analytical value of P(t) for the linear system (c = 0).

    Parameters
    ----------
    t_i           : float – target time
    t_1           : float – initial time
    initial_state : list  – [F(t_1), E(t_1), L(t_1), P(t_1)]
    params        : dict  – system parameters

    Returns
    -------
    float : exact pupae population at t_i
    """
    F1, E1, L1, P1 = initial_state
    dt = t_i - t_1

    k_F = params['delta_F']
    k_E = params['tau_E']  + params['delta_E']
    k_L = params['tau_L']  + params['delta_L'] + 0.01 * L1
    k_P = params['delta_P'] + params['tau_P']

    tau_E = params['tau_E']
    tau_L = params['tau_L']
    beta  = params['beta']

    # Prevent ZeroDivisionError when decay rates coincide
    eps = 1e-10
    if abs(k_E - k_F) < eps: k_E += eps
    if abs(k_L - k_F) < eps: k_L += eps
    if abs(k_L - k_E) < eps: k_L += eps * 2
    if abs(k_P - k_F) < eps: k_P += eps
    if abs(k_P - k_E) < eps: k_P += eps * 2
    if abs(k_P - k_L) < eps: k_P += eps * 3

    # E(t) coefficients
    A_F = (beta * F1) / (k_E - k_F)
    A_E = E1 - A_F

    # L(t) coefficients
    B_F = (tau_E * A_F) / (k_L - k_F)
    B_E = (tau_E * A_E) / (k_L - k_E)
    B_L = L1 - B_F - B_E

    # P(t) coefficients
    C_F = (tau_L * B_F) / (k_P - k_F)
    C_E = (tau_L * B_E) / (k_P - k_E)
    C_L = (tau_L * B_L) / (k_P - k_L)
    C_P = P1 - C_F - C_E - C_L

    return 1/2 * params['tau_P'] * (C_F * np.exp(-k_F * dt) +
            C_E * np.exp(-k_E * dt) +
            C_L * np.exp(-k_L * dt) +
            C_P * np.exp(-k_P * dt))


def pre_release_adult_input(t, transition_time, initial_state, params):
    """
    Adult input generated after the first release by the cohorts present in the
    7D model right before that release.

    `compute_P_exact` returns half of the adult emergence rate, so we convert it
    back to the total flux and split it according to `mu`.
    """
    if transition_time is None or t < transition_time:
        return 0.0, 0.0

    shared_half_flux = compute_P_exact(t, transition_time, initial_state, params)
    total_flux = 2.0 * shared_half_flux

    female_flux = params['mu'] * total_flux
    male_flux = (1.0 - params['mu']) * total_flux
    return female_flux, male_flux

# ─────────────────────────────────────────────────────────────────
# 7-dim population model  (Ms injected analytically)
# ─────────────────────────────────────────────────────────────────

def det_model_7(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, transi,
                death_L, death_P, c, mu,
                t_data, precip_data, H,
                release_times, rho,          # ← release schedule
                type=1):

    E, L, P, F, Ff, Fs, M = y
    Ms = Ms_fun(t, release_times, rho, deltaA)   # pure function call

    if type == 1:
        precip = np.interp(t, t_data, precip_data)
        comp = competition1(1/c, 1/(c * 50), precip)
    elif type == 2:
        water = np.interp(t, t_data, H)
        comp = 1 / ((1 / (10 * c)) * (water / max_rain) + 1/c)
    else:
        comp = c

    if M > 0:
        probaM  = M  / (M + Ms)
        probaMs = Ms / (M + Ms)
    else:
        probaM, probaMs = 0.0, 0.0

    matf = allee(M, Ms) * probaM
    mats = allee(M, Ms) * probaMs

    dE  = n_egg * birth * Ff - death_egg * E - tel * E
    dL  = tel * E - tlp * L - death_L * L - comp * L**2
    dP  = tlp * L - (transi + death_P) * P
    dF  = mu * transi * P - (matf + mats + deltaA) * F
    dFf = matf * F - deltaA * Ff
    dFs = mats * F - deltaA * Fs
    dM  = (1 - mu) * transi * P - deltaA * 3 * M

    return np.array([dE, dL, dP, dF, dFf, dFs, dM])


def sim_7(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, transi_pa,
          death_L, death_P, c, mu,
          release_times=None, rho=0.0,
          n_egg=64, precip_data=precip_data, H=hum_data, type=1):
    """
    Parameters
    ----------
    pop_init      : length 7 (E,L,P,F,Ff,Fs,M) or 8 (Ms0 ignored — use releases)
    release_times : list/array of release times t_i  (None = no releases)
    rho           : scalar or array, release size(s)
    """
    if release_times is None:
        release_times = np.array([])
        rho = 0.0

    y0 = np.asarray(pop_init[:7], dtype=float)

    sol = solve_ivp(
        lambda t, y: det_model_7(
            t, y, birth, n_egg, deltaA, deltaE,
            transi_el, transi_lp, transi_pa,
            death_L, death_P, c, mu,
            time_data, precip_data, H,
            release_times, rho,
            type
        ),
        [days[0], days[-1]], y0,
        t_eval=days, method='LSODA', vectorized=False
    )

    # Reattach Ms for drop-in compatibility with 8-row outputs
    Ms_out = Ms_fun(days, release_times, rho, deltaA)
    return np.vstack([sol.y, Ms_out])


# ─────────────────────────────────────────────────────────────────
# 2-dim reduced model  (Ms injected analytically, same pattern)
# ─────────────────────────────────────────────────────────────────

def det_model_2(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, transi_mod,
                death_L, c,
                t_data, precip_data, H,
                release_times, rho,
                residual_config=None,
                type=1):

    F, M = y
    Ms = Ms_fun(t, release_times, rho, deltaA)   # same call, no ODE

    if type == 1:
        precip = np.interp(t, t_data, precip_data)
        comp = competition1(1/c, 1/(c * 50), precip)
    elif type == 2:
        water = np.interp(t, t_data, H)
        comp = 1 / ((1 / (0.2 * c)) * (water / max_rain) + 1/(5 * c))
    else:
        comp = c

    probaM = M / (M + Ms) if M > 0 else 0.0
    matf   = allee(M, Ms) * probaM

    birth_mod  = birth * tel * n_egg / (tel + death_egg)
    birth_rate = (-(death_L + tlp) + np.sqrt(
                    (death_L + tlp)**2 + matf * F * 4 * comp * birth_mod / (allee(M, Ms) + deltaA)
                 )) / (2 * comp)

    shared = birth_rate * transi_mod / 2

    female_res, male_res = 0.0, 0.0
    if residual_config is not None:
        female_res, male_res = pre_release_adult_input(
            t,
            residual_config['transition_time'],
            residual_config['initial_state'],
            residual_config['params']
        )

    dF = shared * probaM - deltaA * F + female_res
    dM = shared - deltaA * 3 * M + male_res

    return np.array([dF, dM])


def sim_2(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, transi_mod,
          death_L, c,
          release_times=None, rho=0.0,
          n_egg=64, precip_data=precip_data, H=hum_data, type=1,
          pre_release_pop_init_7d=None, transi_pa=None, death_P=None, mu=0.5):
    """
    pop_init               : length 2 (F, M) or 3 (Ms0 ignored — use releases)
    pre_release_pop_init_7d: optional 7D initial state used to bootstrap the
                             reduced model at the first release time
    """
    if release_times is None:
        release_times = np.array([])
        rho = 0.0
    else:
        release_times = np.asarray(release_times, dtype=float)

    y0 = np.asarray(pop_init[:2], dtype=float)
    residual_config = None

    in_window_releases = release_times[
        (release_times >= days[0]) & (release_times <= days[-1])
    ]

    if pre_release_pop_init_7d is not None and in_window_releases.size > 0:
        if transi_pa is None or death_P is None:
            raise ValueError(
                "transi_pa and death_P are required when pre_release_pop_init_7d is provided."
            )

        first_release = float(in_window_releases[0])

        if first_release == days[0]:
            pre_release_state = np.asarray(pre_release_pop_init_7d[:7], dtype=float)
            pre_F = np.array([], dtype=float)
            pre_M = np.array([], dtype=float)
        else:
            pre_days = np.asarray(days[days < first_release], dtype=float)
            pre_t_eval = np.append(pre_days, first_release)

            pre_sol = solve_ivp(
                lambda t, y: det_model_7(
                    t, y, birth, n_egg, deltaA, deltaE,
                    transi_el, transi_lp, transi_pa,
                    death_L, death_P, c, mu,
                    time_data, precip_data, H,
                    np.array([]), 0.0,
                    type
                ),
                [days[0], first_release],
                np.asarray(pre_release_pop_init_7d[:7], dtype=float),
                t_eval=pre_t_eval,
                method='LSODA',
                vectorized=False
            )

            pre_release_state = pre_sol.y[:, -1]
            pre_F = pre_sol.y[3, :-1] + pre_sol.y[4, :-1] + pre_sol.y[5, :-1]
            pre_M = pre_sol.y[6, :-1]

        E0, L0, P0, F0, Ff0, Fs0, M0 = pre_release_state
        y0 = np.array([F0 + Ff0 + Fs0, M0], dtype=float)
        residual_config = {
            'transition_time': first_release,
            'initial_state': [Ff0, E0, L0, P0],
            'params': {
                'delta_F': deltaA,
                'tau_E': transi_el,
                'delta_E': deltaE,
                'tau_L': transi_lp,
                'delta_L': death_L,
                'tau_P': transi_pa,
                'delta_P': death_P,
                'beta': n_egg * birth,
                'mu': mu,
            },
        }

        post_days = np.asarray(days[days >= first_release], dtype=float)

        if post_days.size == 1 and post_days[0] == first_release:
            post_F = np.array([y0[0]])
            post_M = np.array([y0[1]])
        else:
            sol = solve_ivp(
                lambda t, y: det_model_2(
                    t, y, birth, n_egg, deltaA, deltaE,
                    transi_el, transi_lp, transi_mod,
                    death_L, c,
                    time_data, precip_data, H,
                    release_times, rho,
                    residual_config,
                    type
                ),
                [post_days[0], post_days[-1]], y0,
                t_eval=post_days, method='LSODA', vectorized=False
            )
            post_F = sol.y[0]
            post_M = sol.y[1]

        F_out = np.concatenate([pre_F, post_F])
        M_out = np.concatenate([pre_M, post_M])
        Ms_out = Ms_fun(days, release_times, rho, deltaA)
        return np.vstack([F_out, M_out, Ms_out])

    sol = solve_ivp(
        lambda t, y: det_model_2(
            t, y, birth, n_egg, deltaA, deltaE,
            transi_el, transi_lp, transi_mod,
            death_L, c,
            time_data, precip_data, H,
            release_times, rho,
            residual_config,
            type
        ),
        [days[0], days[-1]], y0,
        t_eval=days, method='LSODA', vectorized=False
    )

    # Reattach Ms for drop-in compatibility with 3-row outputs
    Ms_out = Ms_fun(days, release_times, rho, deltaA)
    return np.vstack([sol.y, Ms_out])


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


#  Pre-release adult input 
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


#  Old vs New reduced model comparison 
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
