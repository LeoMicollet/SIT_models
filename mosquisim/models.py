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

    dF = shared * probaM - deltaA * F
    dM = shared - deltaA * 3 * M

    return np.array([dF, dM])


def sim_2(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, transi_mod,
          death_L, c,
          release_times=None, rho=0.0,
          n_egg=64, precip_data=precip_data, H=hum_data, type=1):
    """
    pop_init : length 2 (F, M) or 3 (Ms0 ignored — use releases)
    """
    if release_times is None:
        release_times = np.array([])
        rho = 0.0

    y0 = np.asarray(pop_init[:2], dtype=float)

    sol = solve_ivp(
        lambda t, y: det_model_2(
            t, y, birth, n_egg, deltaA, deltaE,
            transi_el, transi_lp, transi_mod,
            death_L, c,
            time_data, precip_data, H,
            release_times, rho,
            type
        ),
        [days[0], days[-1]], y0,
        t_eval=days, method='LSODA', vectorized=False
    )

    # Reattach Ms for drop-in compatibility with 3-row outputs
    Ms_out = Ms_fun(days, release_times, rho, deltaA)
    return np.vstack([sol.y, Ms_out])