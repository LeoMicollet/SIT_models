# estimation.py
# =============================================================================
# Maximum likelihood estimation for the SIT mechanistic-statistical model.
#
# TWO SETTINGS:
#   1. Simple  — only sterile males captured (wild pop near extinction)
#               Parameters: alpha (capture rate), delta_M (death rate)
#
#   2. Full    — both wild males and females captured (full population)
#               Parameters: alpha_M, alpha_F, A, B, C, delta_M, delta_F
#
# In both cases the observation model is Poisson:
#   N_i ~ Poisson(alpha * C_i)
# where C_i is the integral of the modelled population over week i.
# =============================================================================

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.integrate import solve_ivp


# =============================================================================
# SETTING 1 — SIMPLE (sterile males only, Ms follows analytical formula)
# =============================================================================

def C_Ms(week, releases, delta_M):
    """
    Integral of Ms(t) over week i = [t_{i-1}, t_i], with t in units of weeks
    (each week = 7 days, so times are in days internally).

    C_i = sum_{j<=i-2} r_j/delta * (exp(-delta*7*(i-1-j)) - exp(-delta*7*(i-j)))
        + r_{i-1}/delta * (1 - exp(-delta*7))

    Parameters
    ----------
    week     : int/float  – index of the current week (scalar)
    releases : DataFrame  – columns ['Week', 'Nb_ind']
    delta_M  : float      – sterile male death rate (per day)

    Returns
    -------
    float : expected capturable population during week i
    """
    r_old  = releases.loc[releases["Week"] <= week - 2]
    r_prev = releases.loc[releases["Week"] == week - 1]

    term_old = np.sum(
        r_old["Nb_ind"].to_numpy() * (
            np.exp(-delta_M * 7 * (week - r_old["Week"].to_numpy() - 1))
            - np.exp(-delta_M * 7 * (week - r_old["Week"].to_numpy()))
        )
    )
    term_prev = np.sum(
        r_prev["Nb_ind"].to_numpy() * (
            1 - np.exp(-delta_M * 7)
        )
    )
    return (term_old + term_prev) / delta_M


def negloglik_simple(params, releases, captures, weeks):
    """
    Negative log-likelihood for the simple setting (sterile males only).

    params  = [alpha, delta_M]
    L = sum_i [ c_i*(log(alpha) + log(C_i)) - alpha*C_i ]
    """
    alpha, delta_M = params

    if alpha <= 0 or delta_M <= 0:
        return np.inf

    captures = np.asarray(captures, dtype=float)
    weeks    = np.asarray(weeks,    dtype=float)

    C = np.array([C_Ms(w, releases, delta_M) for w in weeks], dtype=float)
    C = np.clip(C, 1e-12, None)

    ll = np.sum(captures * (np.log(alpha) + np.log(C)) - alpha * C)
    return -ll


def estimate_simple(releases, captures, weeks,
                    alpha0=0.01, delta0=0.139,
                    alpha_bounds=(1e-9, 1.0),
                    delta_bounds=(1e-9, 1.0),
                    tol=1e-12):
    """
    MLE for the simple setting.

    Parameters
    ----------
    releases       : DataFrame – ['Week', 'Nb_ind']
    captures       : array     – observed weekly captures N_i
    weeks          : array     – week indices matching captures
    alpha0, delta0 : floats    – initial guesses
    *_bounds       : tuples    – (min, max) for each parameter
    tol            : float     – optimizer tolerance

    Returns
    -------
    scipy OptimizeResult  (.x = [alpha_hat, delta_hat])
    """
    return opt.minimize(
        negloglik_simple,
        x0=[alpha0, delta0],
        args=(releases, captures, weeks),
        bounds=[alpha_bounds, delta_bounds],
        method="L-BFGS-B",
        tol=tol
    )


def simulate_and_estimate_simple(releases, delta_true, alpha_true, weeks,
                                  n_simulations=50, **kwargs):
    """
    Monte Carlo validation: simulate Poisson captures then re-estimate.

    Returns
    -------
    DataFrame with columns ['alpha', 'delta_M'] — one row per simulation.
    """
    delta_eff = max(delta_true, 1e-12)
    C_true = np.array([C_Ms(w, releases, delta_eff) for w in weeks])

    results = []
    for i in range(n_simulations):
        capt_sim = np.random.poisson(alpha_true * C_true)
        val = estimate_simple(releases, capt_sim, weeks, **kwargs)
        results.append(val.x)
        print(f"[simple] simulation {i+1}/{n_simulations}  → alpha={val.x[0]:.5f}, delta={val.x[1]:.5f}")

    return pd.DataFrame(results, columns=["alpha", "delta_M"])


# =============================================================================
# SETTING 2 — FULL (wild males + females captured, ODE-based population)
# =============================================================================
# The reduced ODE system is:
#   dF/dt = nu     * g(F, M, Ms) - delta_F * F
#   dM/dt = (1-nu) * g(F, M, Ms) - delta_M * M
#
# with g(F, M, Ms) = A * (sqrt(B^2 + C * F_f) - B)
# and  F_f = matf / (matf + mats + delta_F) * F   (fraction of fertile females)
#
# Parameters to estimate: theta = [A, B, C, delta_M, delta_F, alpha_M, alpha_F]
# (nu = sex ratio, fixed at 0.5 unless specified)
# =============================================================================

def Ms_fun_days(t, release_times_days, rho, delta_M):
    """Ms(t) analytical formula in days."""
    release_times_days = np.atleast_1d(np.asarray(release_times_days, dtype=float))
    if np.isscalar(rho):
        rho = np.full(len(release_times_days), float(rho))
    scalar = np.isscalar(t)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    Ms = np.zeros(len(t))
    for i, ti in enumerate(release_times_days):
        mask = t >= ti
        Ms[mask] += rho[i] * np.exp(-delta_M * (t[mask] - ti))
    return float(Ms[0]) if scalar else Ms


def ode_full(t, y, A, B, C, delta_M, delta_F, nu,
             release_times_days, rho):
    """
    RHS of the 2D reduced ODE (F_tot, M).
    g(F, M, Ms) = A*(sqrt(B^2 + C*F) - B)
    Here F is treated as F_tot; the fertile fraction is handled implicitly
    via the QSSA (already encoded in A, B, C).
    """
    F, M = y
    F = max(F, 0.0)
    M = max(M, 0.0)

    Ms = Ms_fun_days(t, release_times_days, rho, delta_M)

    disc = max(B**2 + C * F, 0.0)
    g    = A * (np.sqrt(disc) - B)
    g    = max(g, 0.0)

    dF = nu       * g - delta_F * F
    dM = (1 - nu) * g - delta_M * M
    return [dF, dM]


def integrate_full(days, F0, M0, A, B, C, delta_M, delta_F, nu,
                   release_times_days, rho):
    """
    Solve the ODE and return (F_traj, M_traj) on `days`.
    Splits at release times so LSODA cannot jump over them.
    """
    release_times_days = np.asarray(release_times_days, dtype=float)
    breakpoints = np.concatenate([[days[0]], release_times_days, [days[-1]]])
    breakpoints = np.unique(breakpoints)

    y0 = np.array([F0, M0], dtype=float)
    t_all, y_all = [], []

    for k in range(len(breakpoints) - 1):
        t_start, t_end = breakpoints[k], breakpoints[k+1]
        seg = days[(days >= t_start) & (days <= t_end)] if k == 0 \
         else days[(days >  t_start) & (days <= t_end)]
        if len(seg) == 0:
            continue
        sol = solve_ivp(
            lambda t, y: ode_full(t, y, A, B, C, delta_M, delta_F, nu,
                                  release_times_days, rho),
            [t_start, t_end], y0,
            t_eval=seg, method='LSODA', vectorized=False
        )
        t_all.append(sol.t)
        y_all.append(sol.y)
        y0 = sol.y[:, -1].copy()

    return np.concatenate(t_all), np.hstack(y_all)


def C_population(week_idx, traj_days, traj_pop, week_starts_days):
    """
    Integral of population trajectory over week i = [t_{i-1}, t_i]
    approximated by the trapezoidal rule on the ODE output grid.

    Parameters
    ----------
    week_idx        : int   – index into week_starts_days (0-based)
    traj_days       : array – time points of ODE output (in days)
    traj_pop        : array – population values (F or M) at traj_days
    week_starts_days: array – day of start of each week (length = n_weeks+1)

    Returns
    -------
    float : integral over the week
    """
    t0 = week_starts_days[week_idx]
    t1 = week_starts_days[week_idx + 1]
    mask = (traj_days >= t0) & (traj_days <= t1)
    t_seg = traj_days[mask]
    p_seg = traj_pop[mask]
    if len(t_seg) < 2:
        return 0.0
    return np.trapz(p_seg, t_seg)


def negloglik_full(params, F0, M0, nu,
                   release_times_days, rho,
                   captures_M, captures_F,
                   week_starts_days, sim_days):
    """
    Negative log-likelihood for the full setting.

    params = [log(A), log(B), log(C), log(delta_M), log(delta_F),
              log(alpha_M), log(alpha_F)]
    Log-parameterisation enforces positivity without bounds.

    L = sum_i [n_M*(log(alpha_M)+log(C_M_i)) - alpha_M*C_M_i]
      + sum_i [n_F*(log(alpha_F)+log(C_F_i)) - alpha_F*C_F_i]
    """
    # Unpack in log space (ensures positivity)
    logA, logB, logC, log_dM, log_dF, log_aM, log_aF = params
    A       = np.exp(logA)
    B       = np.exp(logB)
    C       = np.exp(logC)
    delta_M = np.exp(log_dM)
    delta_F = np.exp(log_dF)
    alpha_M = np.exp(log_aM)
    alpha_F = np.exp(log_aF)

    # Integrate ODE
    try:
        t_out, y_out = integrate_full(
            sim_days, F0, M0, A, B, C, delta_M, delta_F, nu,
            release_times_days, rho
        )
    except Exception:
        return np.inf

    F_traj = y_out[0]
    M_traj = y_out[1]

    n_weeks = len(captures_M)
    C_M = np.array([C_population(i, t_out, M_traj, week_starts_days) for i in range(n_weeks)])
    C_F = np.array([C_population(i, t_out, F_traj, week_starts_days) for i in range(n_weeks)])

    C_M = np.clip(C_M, 1e-12, None)
    C_F = np.clip(C_F, 1e-12, None)

    captures_M = np.asarray(captures_M, dtype=float)
    captures_F = np.asarray(captures_F, dtype=float)

    ll = (np.sum(captures_M * (np.log(alpha_M) + np.log(C_M)) - alpha_M * C_M)
        + np.sum(captures_F * (np.log(alpha_F) + np.log(C_F)) - alpha_F * C_F))
    return -ll


def estimate_full(F0, M0, nu,
                  release_times_days, rho,
                  captures_M, captures_F,
                  week_starts_days, sim_days,
                  A0=1.0, B0=1.0, C0=1.0,
                  delta_M0=0.1, delta_F0=0.1,
                  alpha_M0=1e-4, alpha_F0=1e-4,
                  tol=1e-8):
    """
    MLE for the full setting.

    Parameters
    ----------
    F0, M0             : floats  – initial conditions
    nu                 : float   – sex ratio (fraction female, typically 0.5)
    release_times_days : array   – release days
    rho                : float   – release size
    captures_M/F       : arrays  – observed weekly male/female captures
    week_starts_days   : array   – day at start of each week (length = n_weeks+1)
    sim_days           : array   – fine daily grid for ODE integration
    *0                 : floats  – initial guesses for each parameter
    tol                : float   – optimizer tolerance

    Returns
    -------
    dict with keys: A, B, C, delta_M, delta_F, alpha_M, alpha_F, success, message
    """
    x0 = [np.log(A0), np.log(B0), np.log(C0),
          np.log(delta_M0), np.log(delta_F0),
          np.log(alpha_M0), np.log(alpha_F0)]

    result = opt.minimize(
        negloglik_full,
        x0=x0,
        args=(F0, M0, nu,
              release_times_days, rho,
              captures_M, captures_F,
              week_starts_days, sim_days),
        method="Nelder-Mead",   # gradient-free: ODE likelihood is noisy
        options={"maxiter": 10000, "xatol": tol, "fatol": tol}
    )

    estimates = np.exp(result.x)
    return {
        "A"      : estimates[0],
        "B"      : estimates[1],
        "C"      : estimates[2],
        "delta_M": estimates[3],
        "delta_F": estimates[4],
        "alpha_M": estimates[5],
        "alpha_F": estimates[6],
        "success": result.success,
        "message": result.message,
        "negloglik": result.fun
    }


def simulate_and_estimate_full(F0, M0, nu,
                                release_times_days, rho, sim_days,
                                week_starts_days,
                                A_true, B_true, C_true,
                                delta_M_true, delta_F_true,
                                alpha_M_true, alpha_F_true,
                                n_simulations=20, **kwargs):
    """
    Monte Carlo validation for the full setting.

    Returns
    -------
    DataFrame with one row per simulation and columns for each parameter.
    """
    # Generate true trajectories
    t_out, y_out = integrate_full(
        sim_days, F0, M0, A_true, B_true, C_true,
        delta_M_true, delta_F_true, nu,
        release_times_days, rho
    )
    n_weeks = len(week_starts_days) - 1
    C_M_true = np.array([C_population(i, t_out, y_out[1], week_starts_days) for i in range(n_weeks)])
    C_F_true = np.array([C_population(i, t_out, y_out[0], week_starts_days) for i in range(n_weeks)])

    results = []
    for i in range(n_simulations):
        capt_M = np.random.poisson(alpha_M_true * C_M_true)
        capt_F = np.random.poisson(alpha_F_true * C_F_true)

        est = estimate_full(
            F0, M0, nu,
            release_times_days, rho,
            capt_M, capt_F,
            week_starts_days, sim_days,
            **kwargs
        )
        results.append(est)
        print(f"[full] simulation {i+1}/{n_simulations}  success={est['success']}")

    return pd.DataFrame(results)