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
        lambda t, y: det_model_7(
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

def det_model_2(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, transi_mod,
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

    # ── Birth rate (quasi-steady-state larvae) ────────────────────
    birth_mod = birth * tel * n_egg / (tel + death_egg)

    discriminant = (death_L + tlp)**2 + matf * F * 4 * comp * birth_mod / (1 + deltaA)
    discriminant = max(discriminant, 0.0)   # numerical safety

    birth_rate = (-(death_L + tlp) + np.sqrt(discriminant)) / (2 * comp)

    shared = birth_rate * transi_mod / 2
    deltaFup = deltaA #* delta/(deltaA + delta)
    deltaMup = 3 * deltaA #* delta/( 3 * deltaA + delta)

    #shared = birth_rate / 4

    dF = shared - deltaFup * F
    dM = shared - deltaMup *  M

    return np.array([dF, dM])

def sim_2(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp, transi_mod,
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
            lambda t, y: det_model_2(
                t, y, birth, n_egg, deltaA, deltaE,
                transi_el, transi_lp, transi_mod,
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

def det_model_2_corrected(t, y, birth, n_egg, deltaA, death_egg, tel, tlp, transi_mod,
                           death_L, c, epsilon,
                           t_data, precip_data, H,
                           release_times, rho,
                           reltype=1, Sterile=0):
    """
    Reduced model with first-order Fenichel correction.
    The corrected pupal equilibrium is:
        P(t) ≈ P*(Ff) - epsilon * dP*/dFf * dFf/dt
    """

    F, M = y
    F = max(F, 0.0)
    M = max(M, 0.0)

    # ── Ms ────────────────────────────────────────────────────────
    if Sterile == 1:
        release_times = np.atleast_1d(np.asarray(release_times, dtype=float))
        Ms = float(rho) if (len(release_times) > 0 and t >= release_times[0]) else 0.0
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

    # ── Mating probabilities ──────────────────────────────────────
    total = M + Ms
    probaM = M / total if total > 0.0 else 0.0
    matf   = allee(M, Ms) * probaM

    # ── Birth modifier ────────────────────────────────────────────
    birth_mod = birth * tel * n_egg / (tel + death_egg)

    # ── QSSA equilibrium: P*(F) ───────────────────────────────────
    # birth_rate = quasi-steady larvae = L* satisfying comp*L^2 + (dL+tlp)*L - matf*F*birth_mod = 0
    # Then P* = tlp * L* / (transi_pa + death_P)  — but here transi_mod already
    # encodes tlp * transi_pa / ((dL+tlp)(transi_pa+dP)), so:
    #   shared  = birth_rate * transi_mod / 2
    #   P*      = birth_rate * tlp / (transi_pa + death_P)   [not needed explicitly]
    # We only need dP*/dFf which we compute via dL*/dFf below.

    disc  = max((death_L + tlp)**2 + matf * F * 4 * comp * birth_mod / (1 + deltaA), 0.0)
    L_star        = max((-(death_L + tlp) + np.sqrt(disc)) / (2 * comp), 0.0)

    # ── dL*/dF  (analytical derivative of L* w.r.t. F) ───────────
    # L* = [ -(dL+tlp) + sqrt( (dL+tlp)^2 + 4*comp*birth_mod*matf/(1+dA) * F ) ] / (2*comp)
    # d(L*)/dF = [ matf * birth_mod / (1+dA) ] / [ comp * sqrt(disc) ]   if disc > 0
    if disc > 1e-12:
        dLstar_dF = (matf * birth_mod / (1 + deltaA)) / (comp * np.sqrt(disc))
    else:
        dLstar_dF = 0.0

    # P* = tlp * L* / (transi_mod_pa + death_P)
    # Since transi_mod already bundles several rates, we express P* correction
    # directly through L*:  the adult emergence flux = transi_mod * L* / 2
    # => dP*/dF = tlp * dL*/dF / (transi_pa + death_P)
    # But because transi_mod = tlp*transi_pa/((dL+tlp)(transi_pa+dP)) we have:
    #   transi_mod * L* / 2  =  shared  (the emergence flux into F and M)
    # The correction on the flux is transi_mod/2 * dL*/dF * dF/dt
    # so we simply correct birth_rate → birth_rate_corrected:

    # ── dF/dt at zeroth order (needed for the correction) ─────────
    shared_0   = L_star * transi_mod / 2
    dF_dt_0    = shared_0 - deltaA * F

    # ── First-order Fenichel correction on the emergence flux ─────
    # flux_corrected = transi_mod/2 * (L* - epsilon * dL*/dF * dF/dt)
    L_corrected = L_star - epsilon * dLstar_dF * dF_dt_0
    L_corrected = max(L_corrected, 0.0)

    shared = L_corrected * transi_mod / 2

    dF = shared - deltaA * F
    dM = shared - deltaA * 3 * M

    return np.array([dF, dM])

def sim_2_corrected(pop_init, days, birth, deltaA, deltaE, transi_el, transi_lp,
                    transi_mod, death_L, c, epsilon=1.0,
                    release_times=None, rho=0.0,
                    n_egg=64, precip_data=precip_data, H=hum_data,
                    reltype=1, Sterile=0):
    """
    epsilon : timescale separation parameter.
               0   → recovers sim_2 exactly (pure QSSA)
               1   → full first-order Fenichel correction
              0..1 → interpolates between the two
    """
    if release_times is None:
        release_times = np.array([])
        rho = 0.0
    else:
        release_times = np.asarray(release_times, dtype=float)
        release_times = release_times[
            (release_times > days[0]) & (release_times < days[-1])
        ]

    y0 = np.asarray(pop_init[:2], dtype=float)

    breakpoints = np.concatenate([[days[0]], release_times, [days[-1]]])
    t_all, y_all = [], []

    for k in range(len(breakpoints) - 1):
        t_start = breakpoints[k]
        t_end   = breakpoints[k + 1]

        t_seg = days[(days >= t_start) & (days <= t_end)] if k == 0 \
           else days[(days >  t_start) & (days <= t_end)]

        if len(t_seg) == 0:
            continue

        sol = solve_ivp(
            lambda t, y: det_model_2_corrected(
                t, y, birth, n_egg, deltaA, deltaE,
                transi_el, transi_lp, transi_mod,
                death_L, c, epsilon,
                time_data, precip_data, H,
                release_times, rho, reltype, Sterile
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