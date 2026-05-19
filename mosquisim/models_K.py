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

def det_model_K7(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, transi,
                death_L, death_P, c, mu,
                t_data, precip_data, H,
                release_times, rho,          # release schedule
                type=1, Sterile = 0):

    E, L, P, F, Ff, Fs, M = y

    if Sterile == 1:
        Ms = rho if np.any(release_times) and t >= release_times[0] else 0.0
    else:
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

    dE  = n_egg * birth * Ff * (1 - E*comp) - death_egg * E - tel * E
    dL  = tel * E - tlp * L - death_L * L
    dP  = tlp * L - (transi + death_P) * P
    dF  = mu * transi * P - (matf + mats + deltaA) * F
    dFf = matf * F - deltaA * Ff
    dFs = mats * F - deltaA * Fs
    dM  = (1 - mu) * transi * P - deltaA * 3 * M

    return np.array([dE, dL, dP, dF, dFf, dFs, dM])


def sim_K7(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, transi_pa,
          death_L, death_P, c, mu,
          release_times=None, rho=0.0,
          n_egg=64, precip_data=precip_data, H=hum_data, type=1, Sterile = 0):
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
        lambda t, y: det_model_K7(
            t, y, birth, n_egg, deltaA, deltaE,
            transi_el, transi_lp, transi_pa,
            death_L, death_P, c, mu,
            time_data, precip_data, H,
            release_times, rho,
            type, Sterile
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

def det_model_K2(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, tpa, transi_mod,
                death_L, c,
                t_data, precip_data, H,
                release_times, rho,
                reltype=1, Sterile = 0, delta = 0):

    F, M = y
    F = max(F, 0.0)   #  guard against small negative drift from LSODA
    M = max(M, 0.0)
    print(t)

    # ── Ms ────────────────────────────────────────────────────────
    if Sterile == 1:
        Ms = rho if np.any(release_times) and t >= release_times[0] else 0.0
    else:
        Ms = Ms_fun(t, release_times, rho, deltaA)

    # ── Competition ───────────────────────────────────────────────
    if reltype == 1:
        precip = np.interp(t, t_data, precip_data)
        comp = competition1(1/c, 1/(c * 50), precip)
    elif reltype == 2:
        water = np.interp(t, t_data, H)
        comp = 1 / ((1 / (0.2 * c)) * (water / max_rain) + 1/(5 * c))
    else:
        comp = c

    # ── Mating ────────────────────────────────────────────────────
    if M > 0:
        probaM  = M  / (M + Ms)
    else:
        probaM = 0.0

    matf   = allee(M, Ms) * probaM

    # ── Birth rate (quasi-steady-state egg) ────────────────────

    birth_rate = (tel * transi_mod / (tlp + death_L)) * n_egg * (birth**2) * matf * ((F**2 )/ (matf + tpa)) / (n_egg * birth * matf * (F/ (matf + tpa)) * comp + death_egg + tel)
    deltaFup = deltaA * delta/(deltaA + delta)
    deltaMup = 3 * deltaA * delta/( 3 * deltaA + delta)

    shared = birth_rate / 4

    dF = shared - deltaFup * F
    dM = shared - deltaMup * 3 * M

    return np.array([dF, dM])

def sim_K2(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, tpa, transi_mod,
          death_L, c,
          release_times=None, rho=0.0,
          n_egg=64, precip_data=precip_data, H=hum_data, reltype=1, Sterile=0, delta = 0):

    if release_times is None:
        release_times = np.array([])
        rho = 0.0
    else:
        release_times = np.asarray(release_times, dtype=float)
        release_times = release_times[(release_times > days[0]) & (release_times < days[-1])]

    y0 = np.asarray(pop_init[:2], dtype=float)

    # Split integration at each release time so LSODA cannot jump over them
    breakpoints = np.concatenate([[days[0]], release_times, [days[-1]]])

    t_all, y_all = [], []
    for k in range(len(breakpoints) - 1):
        t_start = breakpoints[k]
        t_end   = breakpoints[k + 1]

        # ← only change: strict > for all segments except the first
        if k == 0:
            t_seg = days[(days >= t_start) & (days <= t_end)]
        else:
            t_seg = days[(days > t_start) & (days <= t_end)]

        if len(t_seg) == 0:
            continue

        sol = solve_ivp(
            lambda t, y: det_model_K2(
                t, y, birth, n_egg, deltaA, deltaE,
                transi_el, transi_lp, tpa, transi_mod,
                death_L, c,
                time_data, precip_data, H,
                release_times, rho, reltype, Sterile, delta
            ),
            [t_start, t_end], y0,
            t_eval=t_seg, method='LSODA', vectorized=False
        )

        t_all.append(sol.t)
        y_all.append(sol.y)
        y0 = sol.y[:, -1].copy()

    t_out = np.concatenate(t_all)
    y_out = np.hstack(y_all)

    Ms_out = Ms_fun(t_out, release_times, rho, deltaA)
    return np.vstack([y_out, Ms_out])