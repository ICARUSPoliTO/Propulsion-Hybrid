"""
PyInjection_1D.py
1D model for injector mass flow with N2O.
"""

# ================== ANTI-OVERSUBSCRIPTION ==================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math, argparse
import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================== CONSTANTS / FALLBACK ====================
MU_GAS_FALLBACK = 1.85e-5  # Pa·s (air/N2O gas order of magnitude)
MU_LIQ_FALLBACK = 3.00e-4  # Pa·s (liquid N2O order of magnitude)
AOUT_GAS_FALLBACK = 203.44 # m/s (typical a ~ 200 m/s)

# ================== UTILITY CACHE & STABILIZATION ==================
def _pkey(p, dp=100.0):
    """Quantize P to 100 Pa to increase cache hits and reduce repeated CoolProp calls."""
    return float(round(p/dp)*dp)

@lru_cache(maxsize=16384)  # memoize repeated thermo queries (same p/T)
def _sat_H_rho(fluid, p_key):  # extract enthalpies and densities in two-phase
    p = float(p_key)  # ensure float
    h_f  = cp.PropsSI('H','P',p,'Q',0,fluid)  # sat. liquid enthalpy
    h_g  = cp.PropsSI('H','P',p,'Q',1,fluid)  # sat. vapor enthalpy (query H at given P,Q)
    rho_l= cp.PropsSI('D','P',p,'Q',0,fluid)  # sat. liquid density
    rho_v= cp.PropsSI('D','P',p,'Q',1,fluid)  # sat. vapor density
    return h_f, h_g, rho_l, rho_v

@lru_cache(maxsize=16384)
def _rho_single_T(fluid, p_key, T):  # single-phase density at (P,T)
    p = float(p_key)
    return cp.PropsSI('D','P',p,'T',T,fluid)

def _safe_viscosity(fluid, p, T=None, phase="gas", x=None):
    """
    Robust viscosity:
    - phase == "gas": use (P,T). If T is missing → MU_GAS_FALLBACK.
    - phase == "liq": if T is given use (P,T) on liquid side (guard near saturation);
                      otherwise use (P,Q=0).
    - phase == "two": linear HEM estimate: mu = (1-x)*mu_l + x*mu_g (if x missing → 0.5).
    """
    try:
        if phase == "gas":
            if T is None:
                raise ValueError  # force fallback
            return cp.PropsSI('V', 'P', p, 'T', T, fluid)  # else call CoolProp

        elif phase == "liq":
            if T is not None:
                try:
                    p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)
                    p_eff = max(p, 1.001 * p_sat)    # push to the liquid side
                except Exception:
                    p_eff = p
                return cp.PropsSI('V', 'P', p_eff, 'T', T, fluid)  # CoolProp on liquid side for viscosity
            else:
                return cp.PropsSI('V', 'P', p, 'Q', 0, fluid)  # no T → use saturated liquid

        else:  # "two"
            xx   = 0.5 if (x is None) else float(min(max(x, 0.0), 1.0))  # clamp x to [0,1], default 0.5
            mu_l = _safe_viscosity(fluid, p, T=None, phase="liq")  # use Q=0 to avoid stability issues
            mu_g = _safe_viscosity(fluid, p, T=T,  phase="gas") if T is not None else _safe_viscosity(fluid, p, phase="gas")
            return (1.0 - xx) * mu_l + xx * mu_g  # linear blend as a simple HEM mu_mix

    except Exception:
        return MU_GAS_FALLBACK if phase == "gas" else MU_LIQ_FALLBACK


def _safe_speed_of_sound(fluid, p, T=None, two_phase=False, x=None, rho_l=None, rho_v=None):
    """
    Robust a_out:
    - two_phase: if x, rho_l, rho_v known → Wood's formula (HEM). Otherwise gas a(P, max(T, T_sat+ε)).
    - single-phase gas/liquid: a(P,T) with a small guard near saturation; if T missing → fallback.
    """
    try:
        if two_phase:
            if (x is not None) and (rho_l is not None) and (rho_v is not None) and (rho_l > 0) and (rho_v > 0):
                xx = float(min(max(x, 0.0), 1.0))  # clamp x to [0,1]
                Vv = xx / rho_v                   # vapor volumetric fraction (relative)
                Vl = (1.0 - xx) / rho_l           # liquid volumetric fraction (relative)
                alpha_v = Vv / (Vv + Vl) if (Vv + Vl) > 0 else 0.0  # void fraction of vapor
                alpha_l = 1.0 - alpha_v
                T_sat = cp.PropsSI('T', 'P', p, 'Q', 1, fluid)
                a_g = cp.PropsSI('A', 'P', p, 'T', max(T_sat + 0.5, (T or T_sat + 0.5)), fluid)  # force T on gas side
                try:
                    a_l = cp.PropsSI('A', 'P', max(p, 1.001*cp.PropsSI('P','T', T_sat-0.5, 'Q', 0, fluid)),
                                     'T', T_sat - 0.5, fluid)  # take T below saturation for liquid and P slightly above sat
                except Exception:
                    a_l = cp.PropsSI('A', 'P', p, 'Q', 0, fluid)  # fallback: saturated liquid
                rho_mix = 1.0 / (xx / rho_v + (1.0 - xx) / rho_l)  # HEM mixture density (harmonic mean)
                denom = alpha_l / (rho_l * a_l * a_l) + alpha_v / (rho_v * a_g * a_g)
                if denom > 0:
                    return math.sqrt(1.0 / (rho_mix * denom))  # Wood's formula for two-phase speed of sound
            T_sat = cp.PropsSI('T','P',p,'Q',1,fluid)
            return cp.PropsSI('A','P',p,'T',max(T or T_sat+0.5, T_sat+0.5),fluid)

        if T is None:
            return AOUT_GAS_FALLBACK
        try:
            p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)
            if abs(p - p_sat)/max(p_sat,1.0) < 1e-4:
                return cp.PropsSI('A','P',p,'T',max(T, cp.PropsSI('T','P',p,'Q',1,fluid)+0.5),fluid)
        except Exception:
            pass
        return cp.PropsSI('A','P',p,'T',T,fluid)
    except Exception:
        return AOUT_GAS_FALLBACK


# ================== MIXTURE HELPERS (HEM) ==================
def mixture_rho_HEM(fluid, p, h):
    """Classic HEM: return (rho_m, rho_l, rho_v, x, is_two)."""
    try:
        h_f, h_g, rho_l, rho_v = _sat_H_rho(fluid, _pkey(p))
        # if h < h_f → subcooled liquid; if h > h_g → superheated vapor (not HEM domain, set x=-1)
        x = (h - h_f)/(h_g - h_f) if h_g > h_f else -1.0  # if h_g == h_f (near critical), force x negative
        # soft clamp for small numerical overshoots
        if -1e-8 <= x <= 1.0 + 1e-8:
            x = min(1.0, max(0.0, x))
    except Exception:
        x = -1.0
    if 0.0 <= x <= 1.0:  # two-phase, quality in [0,1]
        rho_m = 1.0 / (x / rho_v + (1.0 - x) / rho_l)  # HEM mixture density as harmonic mean of phase densities
        return rho_m, rho_l, rho_v, x, True  # is_two = True
    return None, None, None, 0.0, False  # single-phase (x=0 by convention), is_two = False

def rho_singlephase_at_T(fluid, p, T, side="gas"):
    # p_safe = _away_from_psat_T(fluid, p, T, rel_eps=1e-4)
    # version with "side" to choose the side w.r.t. p_sat(T)
    try:
        p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)  # vapor saturation pressure at T (Q=1)
        if abs(p - p_sat) / max(p_sat, 1.0) < 1e-4:
            # explicitly choose the side; gas = below p_sat, liq = above p_sat
            p_safe = (0.999 * p_sat) if (side == "gas") else (1.001 * p_sat)
        else:
            p_safe = p
    except Exception:
        # if CoolProp fails, keep p unchanged
        p_safe = p
    return _rho_single_T(fluid, _pkey(p_safe), T)

def nhne_out_alpha_x(fluid, p1, p2, h1, T_line):
    """
    NHNE-style estimate of x and alpha at the outlet (at P2), blending HEM with a weight based on p1:

      - Isenthalpic flash at (P2, H=h1) → x_HEM, rho_l, rho_v (HEM)
      - k_out = sqrt( max(p1-p2,0) / max(pV(T_line)-p2, ε) )
      - Weights: w_HEM = 1/(k_out+1), w_SPI = k_out/(k_out+1)
      - alpha_HEM = (x/ρ_v) / [ (x/ρ_v) + ((1-x)/ρ_l) ]
      - alpha_NHNE = w_HEM * alpha_HEM
      - x_NHNE     = w_HEM * x_HEM

    Note: using p1 in k_out pulls x and alpha toward SPI when p1 is far from pV relative to p2.
    """
    rho_m, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p2, h1)
    if rho_m is None or not is_two:
        # single-phase at outlet → no void, x irrelevant
        return 0.0, 0.0, False

    # alpha_HEM (no-slip) from relative volumetric fractions
    Vdot_v_rel = x / max(rho_v, 1e-12)
    Vdot_l_rel = (1.0 - x) / max(rho_l, 1e-12)
    alpha_HEM  = Vdot_v_rel / max(Vdot_v_rel + Vdot_l_rel, 1e-12)

    # NHNE outlet weight with p_local = p1
    k_out = _k_classic_local(fluid, p1, p2, T_line)
    w_HEM = 1.0 / (k_out + 1.0)

    alpha_NHNE = w_HEM * alpha_HEM
    x_NHNE     = w_HEM * x
    return alpha_NHNE, x_NHNE, True



# ================== ESTIMATE T_out (isenthalpic) ==================
def estimate_T_out_isenthalpic(fluid, p1, T_line, p2):
    """
    Realistic T_out via isenthalpic flash (valve/orifice):
    - compute h_in = h(P1, T_line)
    - at P2:
        * if h_f(P2) <= h_in <= h_g(P2): two-phase → T_out = T_sat(P2)
        * else single-phase → solve T_out from (P2, H=h_in)
    - return (T_out, phase_hint) with phase_hint in {'gas','liq','two_phase'}
    """
    try:
        h_in  = cp.PropsSI('H','P',p1,'T',T_line,fluid)
        T_sat = cp.PropsSI('T','P',p2,'Q',1,fluid)
        h_f2  = cp.PropsSI('H','P',p2,'Q',0,fluid)
        h_g2  = cp.PropsSI('H','P',p2,'Q',1,fluid)
        if h_f2 <= h_in <= h_g2:
            return T_sat, 'two_phase'
        # single-phase: find T from (P2,H=h_in)
        T_out = cp.PropsSI('T','P',p2,'H',h_in,fluid)
        phase = 'gas' if T_out > (T_sat + 0.5) else 'liq'  # practical 0.5 K margin
        return T_out, phase
    except Exception:
        # conservative fallback: no reliable estimate → keep T_line and assume liquid
        return T_line, 'liq'

# ================== FRICTION ==================
def darcy_friction(Re, rel_rough=1e-5):
    if Re < 1e-6:
        return 0.0
    if Re < 1e5:
        return 0.3164 * Re**(-0.25)
    return (-1.8 * math.log10((rel_rough/3.7)**1.11 + 6.9/Re))**(-2)

def darcy_fully_rough(rel_rough=1e-5):
    return (-1.8 * math.log10((rel_rough/3.7)**1.11))**(-2)

# ================== FLOW MODELS (closed-form, 0D) ==================
def solve_mdot_spi(fluid, p1, p2, T_line, D, Cd, phase_out="auto", T_out=None):
    """
    SPI (Single-Phase Incompressible) mass flow as an upper bound:
      m_dot = Cd * A * sqrt(2 * rho * max(p1 - p2, 0))

    Density selection:
      - phase_out="gas":     rho = rho(P2, T_out)  [requires T_out]
      - phase_out="liq":     rho = rho(P2, T_out) if T_out known, else rho(P2, Q=0)
      - phase_out="auto":    if T_out known: gas if T_out > T_sat(P2)+0.5 K, else liq;
                             if T_out missing: try T_line; if unreliable -> liq.
    Note: does not include gas compressibility/choking nor two-phase flashing.
    """
    # ---- sanity checks
    if D <= 0.0 or Cd <= 0.0:
        return 0.0
    if p1 < 0.0 or p2 < 0.0:
        raise ValueError("Pressures must be non-negative (Pa).")

    A  = 0.25 * math.pi * D**2
    dp = p1 - p2
    if dp <= 0.0:
        return 0.0

    # ---- if T_out not provided, estimate it (isenthalpic) for this model
    if T_out is None:
        # isenthalpic estimate is the standard choice for orifices/injectors
        T_est, ph_est = estimate_T_out_isenthalpic(fluid, p1, T_line, p2)
        T_out = T_est
        # if auto, use phase hint; otherwise honor explicit phase_out
        if phase_out == "auto":
            phase_out = 'gas' if ph_est == 'gas' else ('liq' if ph_est == 'liq' else 'liq')  # two_phase → conservative SPI on liquid side

    # ---- decide phase in auto mode, using T_out if available, else T_line as guess
    T_guess = T_out if (T_out is not None) else T_line
    if phase_out == "auto":
        if T_guess is not None:
            try:
                T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
                phase_out = "gas" if (T_guess - T_sat) > 0.5 else "liq"  # practical margin
            except Exception:
                phase_out = "liq"
        else:
            phase_out = "liq"  # conservative and always defined

    # ---- density calculation
    try:
        if phase_out == "gas":
            if T_out is None and T_guess is None:
                # gas requested but no T available
                raise ValueError("phase_out='gas' requires T_out (or at least T_line as guess).")
            rho = rho_singlephase_at_T(fluid, p2, T_out if T_out is not None else T_guess)
        else:  # "liq"
            if T_out is not None:
                # possibly subcooled liquid: use (P2, T_out) with a guard near saturation
                rho = rho_singlephase_at_T(fluid, p2, T_out, side="liq")  # force liquid side
            else:
                # robust fallback: saturated liquid at P2
                rho = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    except Exception:
        # ultimate fallback: saturated liquid at P2 (always defined in valid domain)
        rho = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)

    # ---- SPI mass flow
    return Cd * A * math.sqrt(2.0 * rho * dp)

def solve_mdot_hem(fluid, p1, p2, T_line, D, Cd):
    A = 0.25 * math.pi * D**2
    h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    # isenthalpic flash to determine outlet state (more realistic density)
    try:
        rho_mix = cp.PropsSI('D', 'P', p2, 'H', h1, fluid)
    except Exception:
        # prudent fallback: saturated liquid
        rho_mix = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    deltah = max(h1 - cp.PropsSI('H','P',p2,'Q',0,fluid), 0.0)
    return Cd * A * rho_mix * math.sqrt(2.0 * deltah)

def k_empirical(L, D):
    # NOTE: soft geometric factor max +20% as L/D→0. (Informative; classic NHNE does not use it.)
    L_over_D = L / D
    return 1.0 + 0.2 * math.exp(-L_over_D)

def solve_mdot_nhne(fluid, p1, p2, T_line, D, Cd):
    """
    Classic NHNE (Nowacki–Hottel–Nelson–Eaton, 0D):
      k = sqrt( (p1 - p2) / (pV - p2) ), with pV = P_sat(T_line)

    Robustness notes:
    - if pV <= p2  → no downstream flashing: k -> inf (full weight on SPI)
    - if pV ~ p2   → use epsilon to avoid overflow but keep trend k -> large
    """
    # saturation pressure at T_line (upstream)
    try:
        pV = cp.PropsSI('P', 'T', T_line, 'Q', 1, fluid)
    except Exception:
        pV = p1  # fallback: if unavailable, do not penalize SPI

    # stability epsilon (Pa)
    eps = 50.0e3   # 50 kPa: numerical only, not a physical "fudge"
    den = max(pV - p2, eps)
    num = max(p1 - p2, 0.0)
    k   = math.sqrt(num / den)

    # optional soft clamp: avoid absurd outliers when data is noisy
    k = max(min(k, 6.0), 0.2)
    if den > eps:
        k = math.sqrt(num / den)               # CLASSIC FORMULA
    elif den >= 0.0:
        k = math.sqrt(num / max(den, eps))     # near pV≈p2 → k large but finite
    else:
        k = float('inf')                        # pV < p2 → no flashing: full weight on SPI

    # limit mass flows (0D)
    mdot_spi = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd)
    mdot_hem = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd)

    # NHNE interpolation (with k=inf → 100% SPI)
    if math.isinf(k):
        mdot_nhne = mdot_spi
    else:
        mdot_nhne = (k/(k+1.0))*mdot_spi + (1.0/(k+1.0))*mdot_hem

    return mdot_nhne, k

def _k_classic_local(fluid, p_local, p2, T_line):
    """Classic local NHNE k: k = sqrt( max(p_local-p2,0) / max(pV-p2, ε) )."""
    try:
        pV = cp.PropsSI('P', 'T', T_line, 'Q', 1, fluid)
    except Exception:
        pV = p_local
    eps = 5.0e4  # 50 kPa for numerical stability only
    den = max(pV - p2, eps)
    num = max(p_local - p2, 0.0)
    k = math.sqrt(num/den)
    return k


def mixture_props_NHNE(fluid, p, h1, T_line, p2):
    """
    Local NHNE properties at pressure p:
      - HEM (ρ_m, μ_HEM, x)
      - SPI liquid (ρ_liq, μ_liq)
      - Blending: w_SPI = k/(k+1), w_HEM = 1/(k+1)
    """
    # --- HEM at (p, h1) ---
    rho_m, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p, h1)
    if rho_m is None:
        # single-phase → single-phase fallback
        rho_m = rho_singlephase_at_T(fluid, p, T_line)
        try:
            T_sat = cp.PropsSI('T','P',p,'Q',1,fluid)
            ph = "gas" if (T_line - T_sat) > 0.5 else "liq"
        except Exception:
            ph = "liq"
        mu_HEM = _safe_viscosity(fluid, p, (T_line if ph=="gas" else None), phase=ph)
        x, is_two = 0.0, False
    else:
        # two-phase → linear μ blend
        mu_l = _safe_viscosity(fluid, p, phase="liq")
        mu_g = _safe_viscosity(fluid, p, T_line, phase="gas")
        mu_HEM = (1.0 - x)*mu_l + x*mu_g

    # --- SPI liquid at p ---
    try:
        rho_liq = cp.PropsSI('D','P',p,'Q',0,fluid)
    except Exception:
        rho_liq = rho_singlephase_at_T(fluid, p, T_line, side="liq")
    mu_liq = _safe_viscosity(fluid, p, phase="liq")

    # --- Local NHNE weight ---
    k_loc = _k_classic_local(fluid, p, p2, T_line)
    w_spi = k_loc/(k_loc+1.0)
    w_hem = 1.0/(k_loc+1.0)

    # --- Property blending ---
    inv_rho_eff = w_spi*(1.0/max(rho_liq,1e-12)) + w_hem*(1.0/max(rho_m,1e-12))
    rho_eff = 1.0/max(inv_rho_eff,1e-12)
    mu_eff  = w_spi*mu_liq + w_hem*mu_HEM
    x_eff   = w_hem*x

    return rho_eff, mu_eff, x_eff, is_two

# ================== FULL 1D MODEL ==================
def solve_mdot_1D(fluid, p1, p2, T_line, D, L,
                  K_minor=0.0, rough=1e-5, n_steps=400,
                  f_const=None):
    """
    1D: Darcy–Weisbach + local losses (K_minor), isenthalpic (h=h1),
    NHNE local closure for ρ and μ along the duct.
    """
    A  = 0.25 * math.pi * D**2

    # --- upstream enthalpy (robust choice: sat. liquid at T_line) ---
    USE_H1_SAT_AT_TLINE = True
    if USE_H1_SAT_AT_TLINE:
        h1 = cp.PropsSI('H','T',T_line,'Q',0,fluid)
    else:
        h1 = cp.PropsSI('H','P',p1,'T',T_line,fluid)

    # --- bracketing and tolerances for mdot search ---
    mdot_lo = 1e-6
    mdot_hi = 5.0
    tol_p   = 500.0      # [Pa] tolerance on outlet pressure
    max_it  = 60

    # --- given mdot, integrate p(z) along L and add K_minor → return outlet pressure ---
    def outlet_pressure_for_mdot(mdot):
        G  = mdot / A                # mass flux
        p  = p1
        dz = L / n_steps
        for _ in range(n_steps):
            # local equivalent properties (NHNE blending with local p)
            rho_eff, mu_eff, _, _ = mixture_props_NHNE(fluid, p, h1, T_line, p2)

            # friction factor
            if f_const is not None:
                f = float(f_const)
            else:
                try:
                    U  = G / max(rho_eff, 1e-12)
                    Re = max(rho_eff*U*D / max(mu_eff,1e-12), 1.0)
                    f  = darcy_friction(Re, rough/D)
                except Exception:
                    f  = darcy_fully_rough(rough/D)

            # pressure gradient (Darcy–Weisbach, G-form)
            dp_dz = - (4.0*f/D) * (G*G) / (2.0*max(rho_eff,1e-12))
            p += dp_dz * dz

            # minimum guard on pressure
            if p <= 1e3:
                p = 1e3
                break

        # end local loss (use ρ at the end)
        rho_end, _, _, _ = mixture_props_NHNE(fluid, max(p,1e3), h1, T_line, p2)
        p -= K_minor * (G*G) / (2.0 * max(rho_end, 1e-9))
        return p

    # --- ensure p_out(mdot_lo) and p_out(mdot_hi) bracket p2 ---
    p_out_lo = outlet_pressure_for_mdot(mdot_lo)
    p_out_hi = outlet_pressure_for_mdot(mdot_hi)
    if (p_out_lo - p2) * (p_out_hi - p2) > 0:
        for s in [0.1, 10, 50, 100, 200, 500]:
            mdot_hi = s
            if (p_out_lo - p2) * (outlet_pressure_for_mdot(mdot_hi) - p2) <= 0:
                break

    # --- bisection on mdot: target p_out ≈ p2 ---
    for _ in range(max_it):
        mdot_mid  = 0.5*(mdot_lo + mdot_hi)
        p_out_mid = outlet_pressure_for_mdot(mdot_mid)
        if abs(p_out_mid - p2) < tol_p:
            mdot = mdot_mid
            break
        if (p_out_lo - p2)*(p_out_mid - p2) <= 0:
            mdot_hi = mdot_mid
        else:
            mdot_lo = mdot_mid
    else:
        mdot = mdot_mid  # fallback if not converged early

    # --- outlet state (for diagnostics/BC) ---
    G = mdot / A
    # properties at outlet using local-NHNE (currently collapses to HEM at p2)
    rho2, mu2, x2, is_two2 = mixture_props_NHNE(fluid, p2, h1, T_line, p2)

    # --- NEW: outlet NHNE weighting using p1 (so U_out_1D != U_mix) ---
    # HEM mixture density at (p2, h1) + liquid density (SPI side)
    rho_HEM2, rho_l2, rho_v2, x_HEM2, is_two_hemo = mixture_rho_HEM(fluid, p2, h1)
    if rho_HEM2 is not None:
        try:
            rho_liq2 = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
        except Exception:
            rho_liq2 = rho_singlephase_at_T(fluid, p2, T_line, side="liq")

        # NHNE weight at outlet that “feels” p1 (not p2)
        k_out = _k_classic_local(fluid, p_local=p1, p2=p2, T_line=T_line)
        w_spi = k_out/(k_out + 1.0)
        w_hem = 1.0 /(k_out + 1.0)

        # effective outlet density for U_out_1D (hydraulic)
        inv_rho2 = w_spi*(1.0/max(rho_liq2,1e-12)) + w_hem*(1.0/max(rho_HEM2,1e-12))
        rho2 = 1.0 / max(inv_rho2, 1e-12)
    # --- END NEW ---

    # “exit NHNE” for quality (already present)
    alpha_w, x_w, ok = nhne_out_alpha_x(fluid, p1, p2, h1, T_line)
    if ok:
        x2 = x_w

    U_out = G / max(rho2, 1e-12)


    return dict(
        mdot=mdot, A=A, G=G,
        rho2=rho2, U_out=U_out, x_out=x2,
        rho_l2=None, rho_v2=None, two_phase=is_two2, h1=h1
    )

# ================== POST-PROCESS (state, properties, CFD) ==================
def postprocess_case(fluid, p1_bar, p2_bar, T_line, D, L, Cd, K_minor, rough):
    p1 = float(p1_bar) * 1e5
    p2 = float(p2_bar) * 1e5
    A  = 0.25 * math.pi * D**2

    # ---- 1D (NHNE along the duct)
    res = solve_mdot_1D(fluid, p1, p2, T_line, D, L, K_minor=K_minor, rough=rough, n_steps=400)
    mdot_1D   = res["mdot"]
    G         = res["G"]
    U_out_1D  = res["U_out"]     # velocity to compare with CFD
    h1        = res["h1"]
    x_out     = res["x_out"]     # <-- x already weighted at outlet with p1 (NHNE outlet)

    # ---- Outlet state via isenthalpic flash (for thermo properties)
    T_out = cp.PropsSI('T','P',p2,'H',h1,fluid)
    T_sat = cp.PropsSI('T','P',p2,'Q',1,fluid)
    dT    = T_out - T_sat
    phase_out = "gas" if dT > +1e-6 else ("liquid" if dT < -1e-6 else "two-phase")

    # ---- HEM mixture density at outlet + phase densities (for volumes/CFD)
    rho_mix, rho_l, rho_v, x_out_HEM, is_two = mixture_rho_HEM(fluid, p2, h1)
    if rho_mix is None:
        # Single-phase → choose correct side w.r.t saturation and force coherent x_out
        if dT >= 0.0:  # GAS
            T_eff  = max(T_out, T_sat + 0.5)
            rho_mix = rho_singlephase_at_T(fluid, p2, T_eff, side="gas")
            x_out   = 1.0
            is_two  = False
        else:          # LIQ
            T_eff  = min(T_out, T_sat - 0.5)
            rho_mix = rho_singlephase_at_T(fluid, p2, T_eff, side="liq")
            x_out   = 0.0
            is_two  = False
        # Phase densities helpful for the table
        rho_l = cp.PropsSI('D','P',p2,'Q',0,fluid)
        rho_v = cp.PropsSI('D','P',p2,'Q',1,fluid)

    # ---- Near-saturation blending (HEM ↔ single-phase) ONLY on rho_mix
    DT_blend = 0.8  # [K]
    if is_two:
        try:
            rho_HEM  = rho_mix
            T_ref    = T_sat + 0.5 if dT >= 0.0 else T_sat - 0.5
            side     = "gas" if dT >= 0.0 else "liq"
            rho_mono = rho_singlephase_at_T(fluid, p2, T_ref, side=side)
            w = min(abs(dT) / DT_blend, 1.0)    # 0→HEM; 1→single-phase
            rho_mix = (1.0 - w) * rho_HEM + w * rho_mono
        except Exception:
            pass

    # ---- Mixture viscosity (use final x_out)
    if not is_two:
        mu_mix = _safe_viscosity(fluid, p2, T_out, phase=("gas" if dT >= 0.0 else "liq"))
    else:
        mu_l = _safe_viscosity(fluid, p2, phase="liq")
        mu_g = _safe_viscosity(fluid, p2, T_out, phase="gas")
        mu_mix = (1.0 - x_out)*mu_l + x_out*mu_g

    # ---- Speed of sound and dimensionless numbers (compare with 1D → use U_out_1D)
    a_out   = _safe_speed_of_sound(fluid, p2, T_out, two_phase=is_two, x=x_out, rho_l=rho_l, rho_v=rho_v)
    Mach_1D = U_out_1D / max(a_out, 1e-9)
    Re_1D   = rho_mix * U_out_1D * D / max(mu_mix, 1e-12)

    # ---- 0D comparison models
    mdot_spi  = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd, phase_out="auto", T_out=T_out)
    mdot_hem  = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd)
    mdot_nhne, k_val = solve_mdot_nhne(fluid, p1, p2, T_line, D, Cd)

    # ---- Volumes and superficial velocities (HEM for volumes, with final x_out)
    Vdot   = mdot_1D / rho_mix
    Vdot_v = x_out * mdot_1D / rho_v if (rho_v and x_out > 0) else 0.0
    Vdot_l = (1.0 - x_out) * mdot_1D / rho_l if (rho_l and (1.0 - x_out) > 0) else 0.0
    alpha_out = (Vdot_v / (Vdot_v + Vdot_l)) if (Vdot_v + Vdot_l) > 0 else (1.0 if x_out > 0 else 0.0)
    j_liq = Vdot_l / A
    j_gas = Vdot_v / A

    # Mixture velocity (for BC/phase diagnostics, consistent with j_liq/j_gas)
    U_mix = Vdot / A

    # ---- Phase mass flows (for CFD output)
    mdot_gas = x_out * mdot_1D
    mdot_liq = (1.0 - x_out) * mdot_1D

    # ---- CFD model suggestion
    dT_thresh = 1.0   # [K]
    if (dT >  dT_thresh and not is_two):
        CFD_model = "GAS"
    elif (dT < -dT_thresh and not is_two):
        CFD_model = "LIQ"
    else:
        alpha_lo, alpha_hi = 0.20, 0.80
        if   alpha_out <  alpha_lo: CFD_model = "DPM"
        elif alpha_out <= alpha_hi: CFD_model = "VOF"
        else:                       CFD_model = "GAS+VOF(inj)"

    return {
        # base
        "p1_bar": p1_bar, "A": A,
        "mdot_1D": mdot_1D, "mdot_spi": mdot_spi, "mdot_hem": mdot_hem, "mdot_nhne": mdot_nhne,
        # velocities
        "U_out_1D": U_out_1D,   # FOR CFD COMPARISON
        "U_mix": U_mix,         # FOR BC/DIAGNOSTICS (HEM)
        # outlet state/phases
        "x_out": x_out, "alpha_out": alpha_out,
        "Vdot_l": Vdot_l, "Vdot_v": Vdot_v,
        # CFD ready (use U_out_1D for Mach/Re)
        "T_out": T_out, "T_sat": T_sat, "dT": dT,
        "rho_mix": rho_mix, "mu_mix": mu_mix, "a_out": a_out,
        "Mach": Mach_1D, "Re_out": Re_1D,
        "rho_l": rho_l, "rho_v": rho_v,
        "phase_out": ("two-phase" if is_two else ("gas" if dT >= 0.0 else "liquid")),
        "CFD_model": CFD_model,
        "mdot_liq": mdot_liq, "mdot_gas": mdot_gas,
        "j_liq": j_liq, "j_gas": j_gas,
        # extra monitor
        "k_nhne": k_val
    }

# ================== TABLE PRINTING (ENGLISH VERSION) ==================
def print_table_en(title, columns, rows):
    print(title)
    hdr = " | ".join(f"{h:>{w}}" for (h, _, w, _) in columns)
    sep = "-+-".join("-"*w for (_, _, w, _) in columns)
    print(hdr); print(sep)
    for r in rows:
        cells = []
        for (_, key, w, fmt) in columns:
            val = r.get(key, "")
            if isinstance(val, (int, float)):
                cells.append(f"{val:>{w}{fmt}}")
            else:
                cells.append(f"{str(val):>{w}}")
        print(" | ".join(cells))
    print()

def print_inputs_table_en(params):
    cols = [
        ("Parameter",             "k", 22, "s"),
        ("Value",                 "v", 22, "s"),
    ]
    rows = [
        {"k": "Fluid",              "v": params["fluid"]},
        {"k": "T_line",             "v": f'{params["T_line"]:.3f} K'},
        {"k": "D",                  "v": f'{params["D"]:.6f} m'},
        {"k": "L",                  "v": f'{params["L"]:.6f} m'},
        {"k": "A",                  "v": f'{0.25*math.pi*params["D"]**2:.8f} m^2'},
        {"k": "Cd",                 "v": f'{params["Cd"]:.3f}'},
        {"k": "K_minor (=1/Cd^2)",  "v": f'{(1.0/params["Cd"]**2):.3f}'},
        {"k": "Relative roughness", "v": f'{params["rough"]:.6f}'},
        {"k": "P2 (outlet)",        "v": f'{params["p2_bar"]:.3f} bar'},
    ]
    print_table_en("INITIAL TABLE – Input parameters", cols, rows)

def print_all_tables_en(results):
    # RESULTS (clearly distinguishes both velocities)
    cols_results = [
        ("P1 [bar]",             "p1_bar",      8, ".2f"),
        ("mdot_1D [kg/s]",       "mdot_1D",    15, ".5f"),
        ("mdot_SPI [kg/s]",      "mdot_spi",   15, ".5f"),
        ("mdot_HEM [kg/s]",      "mdot_hem",   15, ".5f"),
        ("mdot_NHNE [kg/s]",     "mdot_nhne",  16, ".5f"),
        ("k_NHNE [-]",           "k_nhne",     12, ".3f"),
        ("x_out [-]",            "x_out",      10, ".4f"),
        ("alpha_out [-]",        "alpha_out",  13, ".4f"),
        ("Vdot_l [m^3/s]",       "Vdot_l",     16, ".5f"),
        ("Vdot_v [m^3/s]",       "Vdot_v",     16, ".5f"),
    ]
    print_table_en(
        "\nRESULT TABLE – Mass flow rates and velocities (U_out 1D = CFD comparison; U_mix HEM = BC/diagnostic)",
        cols_results, results
    )

    # CFD-READY (Mach/Re with U_out_1D)
    cols_cfd = [
        ("P1 [bar]",         "p1_bar",      8,  ".2f"),
        ("T_out [K]",        "T_out",      10,  ".2f"),
        ("T_sat(P2) [K]",    "T_sat",      13,  ".2f"),
        ("ΔT [K]",           "dT",          8,  ".2f"),
        ("rho_mix [kg/m^3]", "rho_mix",    18,  ".3f"),
        ("mu_mix [Pa·s]",    "mu_mix",     16,  ".3e"),
        ("a_out [m/s]",      "a_out",      12,  ".2f"),
        ("Mach (1D) [-]",    "Mach",        12, ".3f"),
        ("Re (1D) [-]",      "Re_out",     13,  ".2e"),
        ("rho_l [kg/m^3]",   "rho_l",      16,  ".3f"),
        ("rho_v [kg/m^3]",   "rho_v",      16,  ".5f"),
    ]
    print_table_en(
        "CFD-READY TABLE – Properties for BC and compressibility check (Mach/Re from U_out 1D)",
        cols_cfd, results
    )

    # PHASES (use U_mix for consistency with j_liq/j_gas and alpha)
    cols_phases = [
        ("P1 [bar]",         "p1_bar",     8,  ".2f"),
        ("phase_out",        "phase_out", 12,  "s"),
        ("CFD_model",        "CFD_model", 12,  "s"),
        ("mdot_liq [kg/s]",  "mdot_liq",  16,  ".5f"),
        ("mdot_gas [kg/s]",  "mdot_gas",  16,  ".5f"),
        ("U_out 1D [m/s]",   "U_out_1D",  14,  ".2f"),
        ("U_mix HEM [m/s]",  "U_mix",     15,  ".2f"),
        ("j_liq [m/s]",      "j_liq",     12,  ".2f"),
        ("j_gas [m/s]",      "j_gas",     12,  ".2f"),
    ]
    print_table_en(
        "PHASE TABLE (for CFD setup) – Phase mass flows and mixture velocity (HEM: U_phases = U_mix)",
        cols_phases, results
    )

    print("LEGEND:")
    print(" - U_out 1D: velocity from 1D model (NHNE + friction) → reference for CFD comparison.")
    print(" - U_mix HEM: average mixture velocity from HEM properties → used for BC and diagnostics.")
    print(" - x_out: mass quality (0=liquid, 0<x<1=two-phase, 1≈vapor).")
    print(" - alpha_out: void fraction (vapor volume fraction).")
    print(" - Vdot_l, Vdot_v: volumetric flow rates of the phases.")
    print(" - Mach (1D), Re (1D): computed using U_out 1D.")
    print()

def print_single_result_en(res):
    """Compact English output for a single P1."""
    print(f"\n=== SINGLE CASE RESULT — P1 = {res['p1_bar']:.2f} bar ===")
    print(f"mdot_1D = {res['mdot_1D']:.6f} kg/s | U_out 1D = {res['U_out_1D']:.2f} m/s "
          f"| Re = {res['Re_out']:.2e} | Mach = {res['Mach']:.3f}")
    print(f"Outlet phase: {res['phase_out']} | x_out = {res['x_out']:.4f} | alpha_out = {res['alpha_out']:.4f}")
    print(f"T_out = {res['T_out']:.2f} K | T_sat = {res['T_sat']:.2f} K | ΔT = {res['dT']:.2f} K")
    print(f"rho_mix = {res['rho_mix']:.3f} kg/m^3 | mu_mix = {res['mu_mix']:.3e} Pa·s | a_out = {res['a_out']:.2f} m/s")
    print(f"mdot_liq = {res['mdot_liq']:.6f} kg/s | mdot_gas = {res['mdot_gas']:.6f} kg/s")
    print(f"j_liq = {res['j_liq']:.3f} m/s | j_gas = {res['j_gas']:.3f} m/s | CFD model: {res['CFD_model']}")
    print(f"k_NHNE = {res['k_nhne']:.3f}\n")


# ================== MAIN (CLI, parallel, plot) ==================
def main():
    parser = argparse.ArgumentParser(description="1D injector model for N2O (CFD-ready tables).")
    parser.add_argument("--p1-start", type=float, help="P1 start [bar]")
    parser.add_argument("--p1-stop",  type=float, help="P1 stop  [bar]")
    parser.add_argument("--p1-step",  type=float, default=1.0, help="P1 step  [bar]")
    parser.add_argument("--p1",       type=float, help="Single P1 [bar] (disables plotting)")
    parser.add_argument("--no-plot",  action="store_true", help="Do not show plot even if sweeping")
    args = parser.parse_args()

    # ---- INPUT ----
    fluid  = "NitrousOxide"
    T_line = 288.0      # K
    D      = 1.5e-3     # m
    L      = 12.5e-3    # m
    Cd     = 0.60
    K_minor= 1.0 / (Cd**2)
    rough  = 1e-5
    p2_bar = 43.0

    # P1 list
    if args.p1 is not None:
        p1_list_bar = [float(args.p1)]
    elif args.p1_start is not None and args.p1_stop is not None:
        p1_list_bar = list(np.arange(float(args.p1_start), float(args.p1_stop)+1e-9, float(args.p1_step)))
    else:
        p1_list_bar = list(np.arange(50.0, 70.0+1e-9, 1.0))  # default sweep

    # Inputs table (English only)
    inputs = dict(fluid=fluid, T_line=T_line, D=D, L=L, Cd=Cd, rough=rough, p2_bar=p2_bar)
    print_inputs_table_en(inputs)

    results = []

    if len(p1_list_bar) == 1:
        # === single case: direct compute, no parallel, no plot ===
        p1b = p1_list_bar[0]
        res = postprocess_case(fluid, p1b, p2_bar, T_line, D, L, Cd, K_minor, rough)
        results.append(res)

        # standard tables (single row) + compact summary
        print_all_tables_en(results)
        print_single_result_en(res)

    else:
        # === sweep: parallel ===
        max_workers = min(24, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut = {
                ex.submit(
                    postprocess_case, fluid, p1b, p2_bar, T_line, D, L, Cd, K_minor, rough
                ): p1b for p1b in p1_list_bar
            }
            for f in as_completed(fut):
                results.append(f.result())

        # sort and print
        results.sort(key=lambda r: r["p1_bar"])
        print_all_tables_en(results)

        # plot only if sweep >= 2 and not disabled
        if (not args.no_plot) and len(results) >= 2:
            p1 = [r["p1_bar"] for r in results]
            plt.figure(figsize=(10,6))
            plt.plot(p1, [r["mdot_1D"]   for r in results], 'o-', label='mdot_1D (NHNE + friction)', linewidth=2)
            plt.plot(p1, [r["mdot_spi"]  for r in results], 's--', label='mdot_SPI (single-phase)', linewidth=2)
            plt.plot(p1, [r["mdot_hem"]  for r in results], 'd--', label='mdot_HEM (equilibrium two-phase)', linewidth=2)
            plt.plot(p1, [r["mdot_nhne"] for r in results], 'x-',  label='mdot_NHNE (empirical)', linewidth=2)
            plt.xlabel('Inlet Pressure $P_1$ [bar]')
            plt.ylabel(r'Mass Flow Rate $\dot{m}$ [kg/s]')
            plt.title('Mass Flow vs Inlet Pressure')
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
