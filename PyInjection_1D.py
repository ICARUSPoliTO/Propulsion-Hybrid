"""
PyInjection_1D.py — 1D model for injector mass flow with N2O.
"""

from __future__ import annotations  # deve stare subito dopo il docstring

# ================== ANTI-OVERSUBSCRIPTION ==================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ================== IMPORT ==================
import math, argparse
import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ================== COSTANTI / FALLBACK ====================
MU_GAS_FALLBACK: float  = 1.85e-5   # Pa·s
MU_LIQ_FALLBACK: float  = 3.00e-4   # Pa·s
AOUT_GAS_FALLBACK: float = 203.44   # m/s

# Tuning opzionale per la dipendenza del blend NHNE dal residence time (L/D)
K_RESIDENCE_COEFF: float = 0.5  # 0=off; tipico 0.2–1.0

# ================== CONFIG TERMODINAMICA ==================
USE_H1_SAT_AT_TLINE = False

# ================== UTILITY CACHE & STABILIZZAZIONE ==================
def _pkey(p: float, dp: float = 100.0) -> float:
    """Pressione quantizzata (Pa) per aumentare i cache hit."""
    return float(round(p / dp) * dp)

@lru_cache(maxsize=16384)
def _sat_H_rho(fluid: str, p_key: float) -> Tuple[float, float, float, float]:
    """(h_f, h_g, rho_l, rho_v) a saturazione alla pressione quantizzata."""
    p = float(p_key)
    h_f  = cp.PropsSI('H', 'P', p, 'Q', 0, fluid)
    h_g  = cp.PropsSI('H', 'P', p, 'Q', 1, fluid)
    rho_l= cp.PropsSI('D', 'P', p, 'Q', 0, fluid)
    rho_v= cp.PropsSI('D', 'P', p, 'Q', 1, fluid)
    return h_f, h_g, rho_l, rho_v

@lru_cache(maxsize=16384)
def _rho_single_T(fluid: str, p_key: float, T: float) -> float:
    """Densità single-phase a (P,T) con P quantizzata."""
    p = float(p_key)
    return cp.PropsSI('D', 'P', p, 'T', T, fluid)

def rho_singlephase_at_T(fluid: str, p: float, T: float, side: str = "gas") -> float:
    """Densità robusta vicino a saturazione: forza 'gas' o 'liq' spostando leggermente P dal Psat(T)."""
    try:
        p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)
        if abs(p - p_sat) / max(p_sat, 1.0) < 1e-4:
            p_safe = (0.999 * p_sat) if (side == "gas") else (1.001 * p_sat)
        else:
            p_safe = p
    except Exception:
        p_safe = p
    return _rho_single_T(fluid, _pkey(p_safe), T)

def _safe_viscosity(fluid: str, p: float, T: Optional[float] = None,
                    phase: str = "gas", x: Optional[float] = None) -> float:
    """Viscosità robusta: gas/liquido singola fase o blend HEM in due-fasi."""
    try:
        if phase == "gas":
            if T is None:
                return MU_GAS_FALLBACK
            return cp.PropsSI('V', 'P', p, 'T', T, fluid)

        if phase == "liq":
            if T is not None:
                try:
                    p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)
                    p_eff = max(p, 1.001 * p_sat)  # forza lato liquido vicino a sat
                except Exception:
                    p_eff = p
                return cp.PropsSI('V', 'P', p_eff, 'T', T, fluid)
            return cp.PropsSI('V', 'P', p, 'Q', 0, fluid)

        # due-fasi: blend lineare HEM
        xx   = 0.5 if (x is None) else float(min(max(x, 0.0), 1.0))
        mu_l = _safe_viscosity(fluid, p, T=None, phase="liq")
        mu_g = _safe_viscosity(fluid, p, T=T, phase="gas") if T is not None else _safe_viscosity(fluid, p, phase="gas")
        return (1.0 - xx) * mu_l + xx * mu_g

    except Exception:
        return MU_GAS_FALLBACK if phase == "gas" else MU_LIQ_FALLBACK


def _safe_speed_of_sound(fluid: str, p: float, T: Optional[float] = None,
                         two_phase: bool = False, x: Optional[float] = None,
                         rho_l: Optional[float] = None, rho_v: Optional[float] = None) -> float:
    """Velocità del suono robusta: Wood in due-fasi, altrimenti a(P,T) con guardie."""
    try:
        if two_phase:
            if (x is not None) and (rho_l and rho_l > 0) and (rho_v and rho_v > 0):
                xx = float(min(max(x, 0.0), 1.0))
                Vv = xx / rho_v
                Vl = (1.0 - xx) / rho_l
                alpha_v = Vv / (Vv + Vl) if (Vv + Vl) > 0 else 0.0
                alpha_l = 1.0 - alpha_v
                T_sat = cp.PropsSI('T', 'P', p, 'Q', 1, fluid)
                a_g = cp.PropsSI('A', 'P', p, 'T', max(T or (T_sat + 0.5), T_sat + 0.5), fluid)
                try:
                    a_l = cp.PropsSI('A', 'P', max(p, 1.001*cp.PropsSI('P','T', T_sat-0.5, 'Q', 0, fluid)),
                                     'T', T_sat - 0.5, fluid)
                except Exception:
                    a_l = cp.PropsSI('A', 'P', p, 'Q', 0, fluid)
                rho_mix = 1.0 / (xx / rho_v + (1.0 - xx) / rho_l)
                denom = alpha_l / (rho_l * a_l * a_l) + alpha_v / (rho_v * a_g * a_g)
                if denom > 0:
                    return math.sqrt(1.0 / (rho_mix * denom))
            # fallback due-fasi → lato gas
            T_sat = cp.PropsSI('T', 'P', p, 'Q', 1, fluid)
            return cp.PropsSI('A', 'P', p, 'T', max(T or (T_sat + 0.5), T_sat + 0.5), fluid)

        if T is None:
            return AOUT_GAS_FALLBACK

        try:
            p_sat = cp.PropsSI('P', 'T', T, 'Q', 1, fluid)
            if abs(p - p_sat) / max(p_sat, 1.0) < 1e-4:
                T_guard = max(T, cp.PropsSI('T', 'P', p, 'Q', 1, fluid) + 0.5)
                return cp.PropsSI('A', 'P', p, 'T', T_guard, fluid)
        except Exception:
            pass

        return cp.PropsSI('A', 'P', p, 'T', T, fluid)

    except Exception:
        return AOUT_GAS_FALLBACK


# ================== MIXTURE HELPERS (HEM / NHNE) ==================
def mixture_rho_HEM(fluid: str, p: float, h: float) -> Tuple[Optional[float], Optional[float],
                                                             Optional[float], float, bool]:
    """HEM a (p,h): qualità x e densità miscela. (rho_m, rho_l, rho_v, x, is_two)."""
    try:
        h_f, h_g, rho_l, rho_v = _sat_H_rho(fluid, _pkey(p))
        x = (h - h_f) / (h_g - h_f) if h_g > h_f else -1.0
        if -1e-8 <= x <= 1.0 + 1e-8:
            x = min(1.0, max(0.0, x))
    except Exception:
        x = -1.0

    if 0.0 <= x <= 1.0:
        rho_m = 1.0 / (x / rho_v + (1.0 - x) / rho_l)
        return rho_m, rho_l, rho_v, x, True

    return None, None, None, 0.0, False

def _k_classic_local(fluid: str, p_local: float, p2: float, T_line: float,
                     L_over_D: Optional[float] = None) -> float:
    """Peso NHNE: k = sqrt(max(p_local-p2,0)/max(pV(T_line)-p2,eps)) con modulazione (1+K*L/D)."""
    try:
        pV = cp.PropsSI('P', 'T', T_line, 'Q', 1, fluid)
    except Exception:
        pV = p_local
    eps = 5.0e4
    den = max(pV - p2, eps)
    num = max(p_local - p2, 0.0)
    k = math.sqrt(num / den)
    if L_over_D is not None and L_over_D > 0.0 and K_RESIDENCE_COEFF > 0.0:
        k /= (1.0 + K_RESIDENCE_COEFF * L_over_D)  # più L/D → più peso a HEM
    return max(0.2, min(k, 6.0))

def nhne_out_alpha_x(fluid: str, p1: float, p2: float, h1: float, T_line: float) -> Tuple[float, float, bool]:
    """Stima NHNE in uscita: alpha e x ottenuti dal ramo HEM pesato con w_HEM=1/(k_out+1)."""
    rho_m, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p2, h1)
    if (rho_m is None) or (not is_two):
        return 0.0, 0.0, False

    Vv = x / max(rho_v, 1e-12)
    Vl = (1.0 - x) / max(rho_l, 1e-12)
    alpha_HEM = Vv / max(Vv + Vl, 1e-12)

    k_out = _k_classic_local(fluid, p1, p2, T_line)
    w_HEM = 1.0 / (k_out + 1.0)

    alpha_NHNE = max(0.0, min(w_HEM * alpha_HEM, 1.0))
    x_NHNE     = max(0.0, min(w_HEM * x,          1.0))
    return alpha_NHNE, x_NHNE, True

# ================== PROPRIETÀ LOCALI (NHNE con scelta SPI) ==================
def mixture_props_NHNE(fluid: str, p: float, h1: float, T_line: float, p2: float,
                       spi_phase_mode: str = "auto") -> Tuple[float, float, float, bool]:
    """Proprietà equivalenti locali via NHNE: blend SPI↔HEM. Ritorna (rho_eff, mu_eff, x_eff, is_two_HEM)."""

    # --- HEM a (p, h1)
    rho_m, _, _, x, is_two = mixture_rho_HEM(fluid, p, h1)
    if rho_m is None:
        # monofase: proprietà coerenti con T_line e lato dedotto da Tsat(p)
        try:
            T_sat_loc = cp.PropsSI('T', 'P', p, 'Q', 1, fluid)
            hem_gas = (T_line - T_sat_loc) > 0.5
        except Exception:
            hem_gas = False
        side = "gas" if hem_gas else "liq"
        rho_m  = rho_singlephase_at_T(fluid, p, T_line, side=side)
        mu_HEM = _safe_viscosity(fluid, p, (T_line if hem_gas else None), phase=("gas" if hem_gas else "liq"))
        x, is_two = 0.0, False
    else:
        # due-fasi: viscosità HEM blenderizzata
        mu_l = _safe_viscosity(fluid, p, phase="liq")
        mu_g = _safe_viscosity(fluid, p, T_line, phase="gas")
        mu_HEM = (1.0 - x) * mu_l + x * mu_g

    # --- SPI: selezione fase (forzata o automatica)
    mode = (spi_phase_mode or "auto").lower()
    if mode not in ("auto", "liq", "gas"):
        mode = "auto"

    if mode == "liq":
        try:
            rho_spi = cp.PropsSI('D', 'P', p, 'Q', 0, fluid)
        except Exception:
            rho_spi = rho_singlephase_at_T(fluid, p, T_line, side="liq")
        mu_spi = _safe_viscosity(fluid, p, phase="liq")

    elif mode == "gas":
        rho_spi = rho_singlephase_at_T(fluid, p, T_line, side="gas")
        mu_spi  = _safe_viscosity(fluid, p, T_line, phase="gas")

    else:  # auto
        if is_two:
            # no-flashing reference → liquido
            try:
                rho_spi = cp.PropsSI('D', 'P', p, 'Q', 0, fluid)
            except Exception:
                rho_spi = rho_singlephase_at_T(fluid, p, T_line, side="liq")
            mu_spi = _safe_viscosity(fluid, p, phase="liq")
        else:
            try:
                T_sat_loc = cp.PropsSI('T', 'P', p, 'Q', 1, fluid)
                is_gas = (T_line - T_sat_loc) > 0.5
            except Exception:
                is_gas = False
            if is_gas:
                rho_spi = rho_singlephase_at_T(fluid, p, T_line, side="gas")
                mu_spi  = _safe_viscosity(fluid, p, T_line, phase="gas")
            else:
                try:
                    rho_spi = cp.PropsSI('D', 'P', p, 'Q', 0, fluid)
                except Exception:
                    rho_spi = rho_singlephase_at_T(fluid, p, T_line, side="liq")
                mu_spi  = _safe_viscosity(fluid, p, phase="liq")

    # --- Pesi NHNE e blend
    k_loc = _k_classic_local(fluid, p_local=p, p2=p2, T_line=T_line)
    w_spi = k_loc / (k_loc + 1.0)
    w_hem = 1.0   / (k_loc + 1.0)

    inv_rho_eff = w_spi * (1.0 / max(rho_spi, 1e-12)) + w_hem * (1.0 / max(rho_m, 1e-12))
    rho_eff = 1.0 / max(inv_rho_eff, 1e-12)
    mu_eff  = w_spi * mu_spi + w_hem * mu_HEM
    x_eff   = w_hem * x  # qualità solo dal ramo HEM

    return rho_eff, mu_eff, x_eff, is_two

# ================== STIMA T_out ===================
def estimate_T_out_energy(fluid: str, p1: float, T_line: float, p2: float,
                          U_out: float, U_in: float = 0.0) -> Tuple[float, float, str]:
    """Stima T_out da bilancio di entalpia di ristagno; ritorna (T_out, h2, phase_hint)."""
    # h1 a monte (fallback a liquido saturo se fallisce T_line)
    try:
        h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    except Exception:
        h1 = cp.PropsSI('H', 'P', p1, 'Q', 0, fluid)

    # h2 = h1 + U_in^2/2 - U_out^2/2 (con guard-rail)
    h2 = h1 + 0.5 * (U_in * U_in - U_out * U_out)
    h2 = max(h2, h1 - 1e7)

    # prova classificazione rispetto alla saturazione a p2
    try:
        T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
        h_f2  = cp.PropsSI('H', 'P', p2, 'Q', 0, fluid)
        h_g2  = cp.PropsSI('H', 'P', p2, 'Q', 1, fluid)
    except Exception:
        T_sat = None
        h_f2 = h_g2 = None

    # two-phase se h2 tra h_f2 e h_g2
    if (h_f2 is not None) and (h_g2 is not None) and (h_f2 <= h2 <= h_g2):
        return (T_sat if T_sat is not None else T_line), h2, 'two_phase'

    # T_out da (P2, H=h2), con hint di fase
    try:
        T_out = cp.PropsSI('T', 'P', p2, 'H', h2, fluid)
    except Exception:
        # fallback: usa T_line come stima
        T_out = T_line

    if T_sat is None:
        phase = 'gas' if T_out >= T_line else 'liq'
    else:
        phase = 'gas' if T_out > (T_sat + 0.5) else 'liq'

    return T_out, h2, phase

# ================== ATTRITO (Darcy–Weisbach) ==================
def darcy_friction(Re: float, rel_rough: float = 1e-5) -> float:
    """Fattore d'attrito: Blasius per Re<1e5, Haaland altrimenti."""
    if Re <= 0.0:
        return 0.0
    if Re < 1.0e5:
        return 0.3164 * Re**(-0.25)
    return (-1.8 * math.log10((rel_rough / 3.7)**1.11 + 6.9 / Re))**-2

def darcy_fully_rough(rel_rough: float = 1e-5) -> float:
    """Limite completamente scabro (Re→∞)."""
    return (-1.8 * math.log10((rel_rough / 3.7)**1.11))**-2

# ================== MODELLI 0D (SPI / HEM / NHNE) ==================
def solve_mdot_spi(fluid: str, p1: float, p2: float, T_line: float,
                   D: float, Cd: float, phase_out: str = "auto",
                   T_out: Optional[float] = None,
                   U_in: float = 0.0,
                   U_out_guess: Optional[float] = None,
                   use_compress: bool = False,
                   n_isentropic: Optional[float] = None) -> float:
    """
    SPI (single-phase incompressible reference, *mai* spento):
    mdot = Cd·A·sqrt(2·rho_liq_ref·Δp).

    NOTE MODELLO (come nel report):
    - Lo SPI rappresenta il limite 'no flashing' e resta *sempre* calcolato, anche se P2 < Psat(T_line).
    - Opzionale correzione di comprimibilità con Y' (Cornelius & Srinivas). Se non si dà n, usiamo K≈ρa².
    - La densità di riferimento è *liquida* a (P1, Q=0): è la curva di riferimento “liquida”.
    """
    if D <= 0.0 or Cd <= 0.0 or p1 <= 0.0 or p2 <= 0.0 or p1 <= p2:
        return 0.0

    A  = 0.25 * math.pi * D * D
    dp = p1 - p2

    # densità di riferimento *liquida* (limite senza flashing)
    try:
        rho_ref = cp.PropsSI('D', 'P', p1, 'Q', 0, fluid)
    except Exception:
        # fallback: (P1, T_line) ma spingendo leggermente lato liquido vicino a sat
        rho_ref = rho_singlephase_at_T(fluid, p1, T_line, side="liq")

    mdot_ideal = Cd * A * math.sqrt(max(2.0 * rho_ref * dp, 0.0))
    if not use_compress:
        return mdot_ideal

    # ---- correzione di comprimibilità Y' ----
    # Se n (isentropico) è fornito → usa forma “ideale gas generalizzato”.
    # Altrimenti stimiamo un rapporto Δp/K, con K≈ρ a^2 (bulk modulus effettivo).
    if n_isentropic is not None and n_isentropic > 1.0:
        pr = max(p2 / p1, 1e-9)
        Yp = (n_isentropic / (n_isentropic - 1.0)) * (1.0 - pr**((n_isentropic - 1.0) / n_isentropic))
    else:
        try:
            a1 = cp.PropsSI('A', 'P', p1, 'T', T_line, fluid)
            K  = max(rho_ref * a1 * a1, 1e5)
        except Exception:
            K  = max(rho_ref * (AOUT_GAS_FALLBACK**2), 1e5)
        Yp = dp / K

    corr = 1.0 / math.sqrt(max(1.0 + Yp, 1e-6))
    return mdot_ideal * corr

def solve_mdot_hem(fluid: str, p1: float, p2: float, T_line: float,
                   D: float, Cd: float,
                   U_in: float = 0.0,
                   U_out_guess: Optional[float] = None) -> float:
    """
    HEM (homogeneous equilibrium) 'energetico' minimale:
    - Non cerca esplicitamente la condizione critica né impone s1=s2 in forma integrale;
      approssima l'equilibrio via h2 ↔ rho_mix(P2,H2) con micro-iterazione.
    - Serve come limite inferiore (flash completo) per il blend NHNE.
    """
    if D <= 0.0 or Cd <= 0.0 or p1 <= 0.0 or p2 <= 0.0 or p1 <= p2:
        return 0.0

    A = 0.25 * math.pi * D * D
    try:
        h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    except Exception:
        return 0.0

    # init con ρ_liq(p2)
    try:
        rho0 = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    except Exception:
        rho0 = max(1.0, cp.PropsSI('D', 'P', p1, 'T', T_line, fluid))

    mdot = Cd * A * math.sqrt(max(2.0 * rho0 * (p1 - p2), 0.0))

    for _ in range(6):
        U_out = mdot / max(rho0 * A, 1e-12)
        h2    = h1 + 0.5 * (U_in**2 - U_out**2)

        try:
            rho_mix = cp.PropsSI('D', 'P', p2, 'H', h2, fluid)
        except Exception:
            rho_mix = rho0

        h_f2 = cp.PropsSI('H', 'P', p2, 'Q', 0, fluid)
        deltah_eff = max(h2 - h_f2, 0.0)

        mdot_new = Cd * A * rho_mix * math.sqrt(max(2.0 * deltah_eff, 0.0))
        if abs(mdot_new - mdot) <= 1e-3 * max(mdot, 1.0):
            mdot = mdot_new
            break
        mdot = mdot_new
        rho0 = rho_mix

    return max(mdot, 0.0)

def solve_mdot_nhne(fluid: str, p1: float, p2: float, T_line: float,
                    D: float, Cd: float,
                    U_in: float = 0.0,
                    U_out_guess: Optional[float] = None,
                    use_spi_compress: bool = False,
                    spi_n: Optional[float] = None,
                    L_over_D: Optional[float] = None) -> Tuple[float, float]:
    """
    NHNE (Dyer): blend continuo tra SPI e HEM
      mdot = (k/(k+1))·mdot_SPI + (1/(k+1))·mdot_HEM
    con k = sqrt((P1-P2) / max(Pv(T_line)-P2, eps)), opzionalmente modulato da L/D.

    NOTE:
    - Niente if di fase: sia SPI che HEM sono sempre calcolati.
    - Se richiesto, lo SPI usa la correzione di comprimibilità (Y').
    """
    # κ classico + eventuale modulazione (L/D) già inglobata in _k_classic_local
    k = _k_classic_local(fluid, p_local=p1, p2=p2, T_line=T_line, L_over_D=L_over_D)

    mdot_spi = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd,
                              phase_out="auto", T_out=None, U_in=U_in, U_out_guess=None,
                              use_compress=use_spi_compress, n_isentropic=spi_n)
    mdot_hem = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd,
                              U_in=U_in, U_out_guess=None)

    mdot_nhne = (k/(k+1.0))*mdot_spi + (1.0/(k+1.0))*mdot_hem
    return mdot_nhne, k


def solve_mdot_1D(fluid: str, p1: float, p2: float, T_line: float,
                  D: float, L: float, K_minor: float = 0.0, rough: float = 1e-5,
                  n_steps: int = 400, f_const: Optional[float] = None,
                  spi_phase_mode: str = "auto",
                  include_accel_loss: bool = True, eta_f: float = 0.0) -> Dict[str, Any]:
    A = 0.25 * math.pi * D * D
    eta_f = float(min(max(eta_f, 0.0), 1.0))

    if USE_H1_SAT_AT_TLINE:
        try:
            h1 = cp.PropsSI('H', 'P', p1, 'Q', 0, fluid)
        except Exception:
            h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    else:
        try:
            h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
        except Exception:
            h1 = cp.PropsSI('H', 'P', p1, 'Q', 0, fluid)

    mdot_lo, mdot_hi = 1e-6, 5.0
    tol_p, max_it = 5.0e2, 60

    def outlet_pressure_for_mdot(mdot: float) -> float:
        if mdot <= 0.0:
            return p1
        G = mdot / A
        p = p1
        dz = L / max(n_steps, 1)

        rho_in, _, _, _ = mixture_props_NHNE(fluid, p1, h1, T_line, p2, spi_phase_mode=spi_phase_mode)
        rho_in = max(rho_in, 1e-12)
        h_loc = h1

        for _ in range(max(n_steps, 1)):
            rho_eff, mu_eff, _, _ = mixture_props_NHNE(fluid, p, h_loc, T_line, p2,
                                                       spi_phase_mode=spi_phase_mode)
            if f_const is not None:
                f = float(f_const)
            else:
                try:
                    U = G / max(rho_eff, 1e-12)
                    Re = max(rho_eff * U * D / max(mu_eff, 1e-12), 1.0)
                    f = darcy_friction(Re, rough / D)
                except Exception:
                    f = darcy_fully_rough(rough / D)

            dp_dz = -(4.0 * f / D) * (G * G) / (2.0 * max(rho_eff, 1e-12))
            dp_step = dp_dz * dz
            p = max(p + dp_step, 1.0e3)

            if eta_f > 0.0:
                h_loc += eta_f * (-dp_step) / max(rho_eff, 1e-12)

            if p <= 1.0e3:
                break

        rho_end, _, _, _ = mixture_props_NHNE(fluid, max(p,1e3), h1, T_line, p2, spi_phase_mode=spi_phase_mode)
        rho_end = max(rho_end, 1.0e-9)

        K_tot = K_minor + max(eta_f, 0.0)
        p -= K_tot * (G * G) / (2.0 * max(rho_end, 1.0e-9))

        if include_accel_loss:
            p -= (G * G) * (1.0 / rho_end - 1.0 / rho_in)

        return p

    p_out_lo = outlet_pressure_for_mdot(mdot_lo)
    p_out_hi = outlet_pressure_for_mdot(mdot_hi)
    if (p_out_lo - p2) * (p_out_hi - p2) > 0.0:
        for s in [0.1, 10, 50, 100, 200, 500]:
            mdot_hi = s
            if (p_out_lo - p2) * (outlet_pressure_for_mdot(mdot_hi) - p2) <= 0.0:
                break

    mdot_mid = 0.5 * (mdot_lo + mdot_hi)
    for _ in range(max_it):
        p_out_mid = outlet_pressure_for_mdot(mdot_mid)
        if abs(p_out_mid - p2) < tol_p:
            mdot = mdot_mid
            break
        if (p_out_lo - p2) * (p_out_mid - p2) <= 0.0:
            mdot_hi = mdot_mid
        else:
            mdot_lo = mdot_mid
        mdot_mid = 0.5 * (mdot_lo + mdot_hi)
    else:
        mdot = mdot_mid

    G  = mdot / A
    dp = max(p1 - p2, 0.0)

    try:
        rho_guess_liq = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    except Exception:
        T_sat_p2 = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
        rho_guess_liq = rho_singlephase_at_T(fluid, p2, max(T_sat_p2 - 0.5, 100.0), side="liq")
    U_out_guess = math.sqrt(2.0 * dp / max(rho_guess_liq, 1e-12))

    T_out_loc = None
    h2_energy = None
    rho2 = None
    x_HEM2 = 0.0
    is_two_hemo = False

    # >>> QUI la riga aggiornata con L/D <<<
    k_out = _k_classic_local(fluid, p_local=p1, p2=p2, T_line=T_line,
                             L_over_D=(L / max(D,1e-12)))
    w_spi = k_out / (k_out + 1.0)
    w_hem = 1.0   / (k_out + 1.0)

    for _ in range(2):
        T_out_loc, h2_energy, phase_hint = estimate_T_out_energy(
            fluid, p1, T_line, p2, U_out=(U_out_guess if rho2 is None else G / max(rho2, 1e-12)), U_in=0.0
        )
        rho_HEM2, _, _, x_HEM2, is_two_hemo = mixture_rho_HEM(fluid, p2, h2_energy)

        if   spi_phase_mode == "gas": spi_is_gas = True
        elif spi_phase_mode == "liq": spi_is_gas = False
        else:                         spi_is_gas = (phase_hint == 'gas')

        try:
            if spi_is_gas:
                T_sat_loc = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
                rho_spi2 = rho_singlephase_at_T(fluid, p2, max(T_out_loc, T_sat_loc + 0.5), side="gas")
            else:
                rho_spi2 = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
        except Exception:
            rho_spi2 = rho_singlephase_at_T(fluid, p2, T_out_loc, side=("gas" if spi_is_gas else "liq"))

        if rho_HEM2 is not None:
            inv_rho2 = w_spi * (1.0 / max(rho_spi2, 1e-12)) + w_hem * (1.0 / max(rho_HEM2, 1e-12))
            rho2 = 1.0 / max(inv_rho2, 1.0e-12)
        else:
            rho2, _, _, _ = mixture_props_NHNE(fluid, p2, h1, T_line, p2, spi_phase_mode=spi_phase_mode)

        U_out_guess = G / max(rho2, 1.0e-12)

    if rho_HEM2 is not None:
        x_spi = 1.0 if spi_is_gas else 0.0
        x_out = w_hem * float(max(min(x_HEM2, 1.0), 0.0)) + w_spi * x_spi
    else:
        _, _, x_tmp, _ = mixture_props_NHNE(fluid, p2, h1, T_line, p2, spi_phase_mode=spi_phase_mode)
        x_out = float(max(min(x_tmp, 1.0), 0.0))

    U_out = G / max(rho2, 1.0e-12)

    return dict(
        mdot=mdot, A=A, G=G,
        rho2=rho2, U_out=U_out, x_out=x_out,
        rho_l2=None, rho_v2=None, two_phase=is_two_hemo, h1=h1
    )


# ================== WRAPPER MULTI-FORO (plain orifice) ==================
def run_plain_orifice_case(fluid: str, p1_bar: float, p2_bar: float, T_line: float,
                           D_input: float, L: float, Cd: float, K_minor: float, rough: float,
                           n_holes: int = 1, keep_total_area: bool = True,
                           include_accel_loss: bool = True, eta_f: float = 0.0,
                           use_spi_compress: bool = False, spi_n: Optional[float] = None) -> dict:
    n = max(1, int(n_holes))
    if n == 1:
        res = postprocess_case(
            fluid=fluid,
            p1_bar=p1_bar,
            p2_bar=p2_bar,
            T_line=T_line,
            D=D_input,
            L=L,
            Cd=Cd,
            K_minor=K_minor,
            rough=rough,
            include_accel_loss=include_accel_loss,
            eta_f=eta_f,
            use_spi_compress=use_spi_compress,
            spi_n=spi_n
        )
        res.update(dict(n_holes=1, D_per_hole=D_input, A_total=res["A"], A_per_hole=res["A"]))
        return res

    D_hole = (D_input / math.sqrt(n)) if keep_total_area else D_input
    base = postprocess_case(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D_hole,
        L=L,
        Cd=Cd,
        K_minor=K_minor,
        rough=rough,
        include_accel_loss=include_accel_loss,
        eta_f=eta_f,
        use_spi_compress=use_spi_compress,
        spi_n=spi_n
    )
    res = base.copy()
    for k in ["mdot_1D", "mdot_spi", "mdot_hem", "mdot_nhne", "Vdot_l", "Vdot_v", "mdot_liq", "mdot_gas"]:
        res[k] *= n
    res["A"] *= n
    res.update(dict(n_holes=n, D_per_hole=D_hole, A_total=res["A"], A_per_hole=res["A"]/n))
    return res
# ================== PRESSURE-SWIRL EMPIRICAL PACK ==================
def estimate_pressure_swirl_params(fluid: str,
                                   p1: float, p2: float, T_line: float, d_orif: float,
                                   mdot_1D: float,
                                   aircore_factor: float = 0.45,
                                   theta_default_deg: float = 55.0,
                                   Csmd: float = 2.25,
                                   dist_type: str = "Rosin-Rammler",
                                   q_rr: float = 3.5,
                                   swirl_profile: str = "free") -> Dict[str, Any]:
    """Stime empiriche per pressure-swirl: (θ, A_eff, U0, SMD, distribuzione, swirl number).
    
    swirl_profile:
      - "free"  → vortice libero nell’anello (S ≈ tan θ)
      - "solid" → rotazione solida (S ≈ (2/3) tan θ)
    """
    d = float(d_orif)
    A_orif = 0.25 * math.pi * d**2
    phi = min(max(float(aircore_factor), 0.0), 0.99)  # clamp φ
    A_eff = max(phi * A_orif, 1e-3 * A_orif)

    # stato di riferimento (lato liquido)
    try:
        T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
    except Exception:
        T_sat = T_line

    try:
        rho_l = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    except Exception:
        rho_l = rho_singlephase_at_T(fluid, p2, max(T_sat - 0.5, 100.0), side="liq")

    mu_l = _safe_viscosity(fluid, p2, T=max(T_sat - 0.5, 100.0), phase="liq")

    try:
        sigma = cp.PropsSI('I', 'P', p2, 'Q', 0, fluid)
    except Exception:
        sigma = 0.010  # N/m

    # idraulica base + SMD Lefebvre-like
    dP = max(p1 - p2, 0.0)
    U0 = mdot_1D / max(rho_l * A_eff, 1e-12)   # velocità media assiale nell’anello
    theta_deg = float(theta_default_deg)

    SMD = Csmd * ((mu_l**0.25) * (sigma**0.25)) / ((rho_l**0.25) * (max(dP, 1.0)**0.5))
    SMD *= max(d, 1e-6)**0.25
    SMD = min(max(SMD, 5e-6), 300e-6)

    # === Swirl ratio & Swirl number ===
    theta_rad = math.radians(theta_deg)
    swirl_ratio = math.tan(theta_rad)  # U_t / U_x

    prof = (swirl_profile or "free").lower()
    if prof not in ("free", "solid"):
        prof = "free"

    if prof == "free":
        swirl_number = swirl_ratio                     # S ≈ tan θ (vortice libero + Ux uniforme)
    else:
        swirl_number = (2.0 / 3.0) * swirl_ratio       # S ≈ (2/3) tan θ (rotazione solida)

    return dict(
        theta_deg=theta_deg,
        A_eff=A_eff,
        U0=U0,
        SMD=SMD,
        d32=SMD,
        dist=dist_type,
        q_rr=q_rr,
        mdot=mdot_1D,
        swirl_ratio=swirl_ratio,
        swirl_number=swirl_number,
        swirl_profile=prof,
        notes="Tune aircore_factor, theta_deg, Csmd se disponi di dati specifici. S calcolato da θ."
    )


# ================== POST-PROCESS (stato uscita, grandezze per CFD) ==================
def postprocess_case(fluid: str, p1_bar: float, p2_bar: float, T_line: float,
                     D: float, L: float, Cd: float, K_minor: float, rough: float,
                     include_accel_loss: bool = True, eta_f: float = 0.0,
                     use_spi_compress: bool = False, spi_n: Optional[float] = None) -> dict:
    p1 = float(p1_bar) * 1e5
    p2 = float(p2_bar) * 1e5
    A  = 0.25 * math.pi * D**2

    # rapporto d’isolamento P2/Psat(T_line)
    try:
        p_sat_line = cp.PropsSI('P','T',T_line,'Q',1,fluid)
        iso_ratio = float(p2 / max(p_sat_line, 1.0))
    except Exception:
        iso_ratio = float('nan')

    # 1D
    res1D = solve_mdot_1D(fluid, p1, p2, T_line, D, L,
                        K_minor=K_minor, rough=rough, n_steps=400,
                        include_accel_loss=include_accel_loss, eta_f=eta_f)
    mdot_1D  = res1D["mdot"]
    U_out_1D = res1D["U_out"]
    h1       = res1D["h1"]
    x_out    = res1D["x_out"]

    # flash (P2,H=h1)
    try:
        T_out = cp.PropsSI('T','P',p2,'H',h1,fluid)
    except Exception:
        T_out = T_line
    try:
        T_sat = cp.PropsSI('T','P',p2,'Q',1,fluid)
    except Exception:
        T_sat = T_out
    dT = T_out - T_sat

    rho_mix, rho_l, rho_v, _, is_two = mixture_rho_HEM(fluid, p2, h1)
    if rho_mix is None:
        if dT >= 0.0:
            T_eff  = max(T_out, T_sat + 0.5)
            rho_mix = rho_singlephase_at_T(fluid, p2, T_eff, side="gas")
            x_out, is_two = 1.0, False
        else:
            T_eff  = min(T_out, T_sat - 0.5)
            rho_mix = rho_singlephase_at_T(fluid, p2, T_eff, side="liq")
            x_out, is_two = 0.0, False
        try:
            rho_l = cp.PropsSI('D','P',p2,'Q',0,fluid)
            rho_v = cp.PropsSI('D','P',p2,'Q',1,fluid)
        except Exception:
            pass

    if is_two:
        try:
            DT_blend = 0.8
            T_ref  = T_sat + 0.5 if dT >= 0.0 else T_sat - 0.5
            side   = "gas" if dT >= 0.0 else "liq"
            rho_mono = rho_singlephase_at_T(fluid, p2, T_ref, side=side)
            w = min(abs(dT)/DT_blend, 1.0)
            rho_mix = (1.0 - w)*rho_mix + w*rho_mono
        except Exception:
            pass

    if not is_two:
        mu_mix = _safe_viscosity(fluid, p2, T_out, phase=("gas" if dT >= 0.0 else "liq"))
    else:
        mu_l = _safe_viscosity(fluid, p2, phase="liq")
        mu_g = _safe_viscosity(fluid, p2, T_out, phase="gas")
        mu_mix = (1.0 - x_out)*mu_l + x_out*mu_g

    a_out   = _safe_speed_of_sound(fluid, p2, T_out, two_phase=is_two, x=x_out, rho_l=rho_l, rho_v=rho_v)
    Mach_1D = U_out_1D / max(a_out, 1e-9)
    Re_1D   = rho_mix * U_out_1D * D / max(mu_mix, 1e-12)

    # 0D
    mdot_spi  = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd,
                            phase_out="auto", T_out=None, U_in=0.0, U_out_guess=None,
                            use_compress=use_spi_compress, n_isentropic=spi_n)

    mdot_hem  = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd, U_in=0.0, U_out_guess=None)

    mdot_nhne, kv = solve_mdot_nhne(fluid, p1, p2, T_line, D, Cd,
                                    U_in=0.0, U_out_guess=None,
                                    use_spi_compress=use_spi_compress, spi_n=spi_n,
                                    L_over_D=(L / max(D, 1e-12)))


    Vdot   = mdot_1D / max(rho_mix, 1e-12)
    Vdot_v = x_out * mdot_1D / rho_v if (rho_v and x_out > 0.0) else 0.0
    Vdot_l = (1.0 - x_out) * mdot_1D / rho_l if (rho_l and (1.0 - x_out) > 0.0) else 0.0
    alpha_out = (Vdot_v / (Vdot_v + Vdot_l)) if (Vdot_v + Vdot_l) > 0.0 else (1.0 if x_out > 0 else 0.0)
    alpha_out = min(max(alpha_out, 0.0), 1.0)
    j_liq = Vdot_l / A
    j_gas = Vdot_v / A
    U_mix = Vdot / A

    dT_thr = 1.0
    if (dT > dT_thr and not is_two):
        CFD_model = "GAS"
    elif (dT < -dT_thr and not is_two):
        CFD_model = "LIQ"
    else:
        if   alpha_out < 0.20: CFD_model = "DPM"
        elif alpha_out <= 0.80: CFD_model = "VOF"
        else:                   CFD_model = "GAS+VOF(inj)"

    warnings = []
    if Mach_1D > 0.9:
        warnings.append(f"High Mach at outlet (M={Mach_1D:.2f}). Possible choking.")
    try:
        Tc = cp.PropsSI('Tcrit', fluid)
        Pc = cp.PropsSI('pcrit', fluid)
        if (abs(T_out - Tc) < 2.0) or (abs(p2 - Pc)/Pc < 0.03):
            warnings.append("Near-critical region: properties may be stiff; treat results with caution.")
    except Exception:
        pass
    if iso_ratio == iso_ratio and iso_ratio <= 0.80:
        warnings.append(f"Isolating regime likely (P2/Psat(T_line) = {iso_ratio:.2f}).")

    return dict(
        p1_bar=p1_bar, A=A,
        mdot_1D=mdot_1D, mdot_spi=mdot_spi, mdot_hem=mdot_hem, mdot_nhne=mdot_nhne,
        U_out_1D=U_out_1D, U_mix=U_mix,
        x_out=x_out, alpha_out=alpha_out, Vdot_l=Vdot_l, Vdot_v=Vdot_v,
        T_out=T_out, T_sat=T_sat, dT=dT, rho_mix=rho_mix, mu_mix=mu_mix,
        a_out=a_out, Mach=Mach_1D, Re_out=Re_1D,
        rho_l=rho_l, rho_v=rho_v,
        phase_out=("two-phase" if is_two else ("gas" if dT >= 0.0 else "liquid")),
        CFD_model=CFD_model,
        mdot_liq=(1.0 - x_out) * mdot_1D, mdot_gas=x_out * mdot_1D,
        j_liq=j_liq, j_gas=j_gas,
        k_nhne=kv,
        iso_ratio=iso_ratio,
        warnings=warnings
    )

# ================== STAMPA TABELLE (EN) ==================
def print_table_en(title: str, columns: list, rows: list) -> None:
    """Stampa tabella semplice con allineamento a larghezze fisse."""
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


def print_inputs_table_en(params: dict) -> None:
    cols = [
        ("Parameter", "k", 22, "s"),
        ("Value",     "v", 22, "s"),
    ]
    rows = [
        {"k": "Injector type",      "v": params.get("injector", "plain")},
        {"k": "Fluid",              "v": params["fluid"]},
        {"k": "T_line",             "v": f'{params["T_line"]:.3f} K'},
        {"k": "D (input)",          "v": f'{params["D"]:.6f} m'},
        {"k": "L",                  "v": f'{params["L"]:.6f} m'},
        {"k": "A (from D)",         "v": f'{0.25*math.pi*params["D"]**2:.8f} m^2'},
        {"k": "Cd",                 "v": f'{params["Cd"]:.3f}'},
        {"k": "K_minor (=1/Cd^2)",  "v": f'{(1.0/params["Cd"]**2):.3f}'},
        {"k": "Relative roughness", "v": f'{params["rough"]:.6f}'},
        {"k": "P2 (outlet)",        "v": f'{params["p2_bar"]:.3f} bar'},
    ]
    if params.get("injector", "plain") == "plain":
        rows.insert(4, {"k": "n_holes",         "v": f'{params.get("n_holes",1)}'})
        rows.insert(5, {"k": "Keep total area", "v": str(params.get("keep_total_area", True))})
    print_table_en("INITIAL TABLE – Input parameters", cols, rows)


def print_all_tables_en(results: list) -> None:
    cols_results = [
        ("P1 [bar]",        "p1_bar",      8, ".2f"),
        ("mdot_1D [kg/s]",  "mdot_1D",    15, ".5f"),
        ("mdot_SPI [kg/s]", "mdot_spi",   15, ".5f"),
        ("mdot_HEM [kg/s]", "mdot_hem",   15, ".5f"),
        ("mdot_NHNE [kg/s]","mdot_nhne",  16, ".5f"),
        ("k_NHNE [-]",      "k_nhne",     12, ".3f"),
        ("x_out [-]",       "x_out",      10, ".4f"),
        ("alpha_out [-]",   "alpha_out",  13, ".4f"),
        ("Vdot_l [m^3/s]",  "Vdot_l",     16, ".5f"),
        ("Vdot_v [m^3/s]",  "Vdot_v",     16, ".5f"),
    ]
    print_table_en(
        "\nRESULT TABLE – Mass flow rates and velocities (U_out 1D = CFD comparison; U_mix HEM = BC/diagnostic)",
        cols_results, results
    )

    cols_cfd = [
        ("P1 [bar]",         "p1_bar",      8,  ".2f"),
        ("T_out [K]",        "T_out",      10,  ".2f"),
        ("T_sat(P2) [K]",    "T_sat",      13,  ".2f"),
        ("ΔT [K]",           "dT",          8,  ".2f"),
        ("P2/Psat(T1) [-]",  "iso_ratio",  14,  ".2f"),
        ("rho_mix [kg/m^3]", "rho_mix",    18,  ".3f"),
        ("mu_mix [Pa·s]",    "mu_mix",     16,  ".3e"),
        ("a_out [m/s]",      "a_out",      12,  ".2f"),
        ("Mach (1D) [-]",    "Mach",       12,  ".3f"),
        ("Re (1D) [-]",      "Re_out",     13,  ".2e"),
        ("rho_l [kg/m^3]",   "rho_l",      16,  ".3f"),
        ("rho_v [kg/m^3]",   "rho_v",      16,  ".5f"),
    ]
    print_table_en(
        "CFD-READY TABLE – Properties for BC and compressibility check (Mach/Re from U_out 1D)",
        cols_cfd, results
    )

    cols_phases = [
        ("P1 [bar]",        "p1_bar",     8,  ".2f"),
        ("phase_out",       "phase_out", 12,  "s"),
        ("CFD_model",       "CFD_model", 12,  "s"),
        ("mdot_liq [kg/s]", "mdot_liq",  16,  ".5f"),
        ("mdot_gas [kg/s]", "mdot_gas",  16,  ".5f"),
        ("U_out 1D [m/s]",  "U_out_1D",  14,  ".2f"),
        ("U_mix HEM [m/s]", "U_mix",     15,  ".2f"),
        ("j_liq [m/s]",     "j_liq",     12,  ".2f"),
        ("j_gas [m/s]",     "j_gas",     12,  ".2f"),
    ]
    print_table_en(
        "PHASE TABLE (for CFD setup) – Phase mass flows and mixture velocity (HEM: U_phases = U_mix)",
        cols_phases, results
    )

    print("LEGEND:")
    print(" - U_out 1D: velocità dal modello 1D (NHNE + attrito) → riferimento confronto CFD.")
    print(" - U_mix HEM: velocità media miscela (HEM) → BC/diagnostica.")
    print(" - x_out: qualità in massa (0=liq, 0<x<1=due-fasi, 1≈vapore).")
    print(" - alpha_out: frazione volumetrica di vapore.")
    print(" - Vdot_l, Vdot_v: portate volumetriche delle fasi.")
    print(" - Mach/Re calcolati con U_out 1D.\n")

def print_single_result_en(res: dict) -> None:
    print(f"\n=== SINGLE CASE RESULT — P1 = {res['p1_bar']:.2f} bar ===")
    print(f"mdot_1D = {res['mdot_1D']:.6f} kg/s | U_out 1D = {res['U_out_1D']:.2f} m/s "
          f"| Re = {res['Re_out']:.2e} | Mach = {res['Mach']:.3f}")
    print(f"Outlet phase: {res['phase_out']} | x_out = {res['x_out']:.4f} | alpha_out = {res['alpha_out']:.4f}")
    print(f"T_out = {res['T_out']:.2f} K | T_sat = {res['T_sat']:.2f} K | ΔT = {res['dT']:.2f} K")
    print(f"P2/Psat(T_line) = {res.get('iso_ratio', float('nan')):.2f}")
    print(f"rho_mix = {res['rho_mix']:.3f} kg/m^3 | mu_mix = {res['mu_mix']:.3e} Pa·s | a_out = {res['a_out']:.2f} m/s")
    print(f"mdot_liq = {res['mdot_liq']:.6f} kg/s | mdot_gas = {res['mdot_gas']:.6f} kg/s")
    print(f"j_liq = {res['j_liq']:.3f} m/s | j_gas = {res['j_gas']:.3f} m/s | CFD model: {res['CFD_model']}")
    print(f"k_NHNE = {res['k_nhne']:.3f}")
    if res.get("warnings"):
        print("Warnings:")
        for w in res["warnings"]:
            print(" -", w)
    print()


def print_swirler_table_en(slist: list) -> None:
    """Tabella parametri empirici pressure-swirl."""
    cols = [
        ("P1 [bar]",        "p1_bar",       8,  ".2f"),
        ("mdot_1D [kg/s]",  "mdot",        14,  ".5f"),
        ("Half-cone [deg]", "theta_deg",   15,  ".1f"),
        ("A_eff [m^2]",     "A_eff",       14,  ".3e"),
        ("U0 [m/s]",        "U0",          10,  ".2f"),
        ("SMD d32 [μm]",    "SMD_um",      14,  ".1f"),
        ("Swirl S [-]",     "S",           12,  ".3f"),
        ("Ut/Ux [-]",       "SR",          10,  ".3f"),
        ("Dist.",           "dist",        10,  "s"),
        ("q (RR)",          "q_rr",         8,  ".1f"),
        ("φ_aircore [-]",   "phi_aircore", 14,  ".2f"),
        ("Profile",         "profile",     10,  "s"),
    ]
    rows = []
    for s in slist:
        rows.append({
            "p1_bar": s["p1_bar"], "mdot": s["mdot"],
            "theta_deg": s["theta_deg"], "A_eff": s["A_eff"], "U0": s["U0"],
            "SMD_um": s["SMD"] * 1e6,
            "S": s.get("swirl_number", float("nan")),
            "SR": s.get("swirl_ratio", float("nan")),
            "dist": s["dist"], "q_rr": s["q_rr"],
            "phi_aircore": s["A_eff"] / max(0.25 * math.pi * s["D"]**2, 1e-20),
            "profile": s.get("swirl_profile", "free"),
        })
    print_table_en("\nPRESSURE-SWIRL (CFD injector inputs — empirical)", cols, rows)

# ================== CONFIG PER I WORKER (picklable) ==================
@dataclass
class RunConfig:
    injector: str
    n_holes: int
    keep_total_area: bool
    swirl_aircore: float
    swirl_theta: float
    swirl_Csmd: float
    swirl_qrr: float
    fluid: str
    T_line: float
    D: float
    L: float
    Cd: float
    K_minor: float
    rough: float
    p2_bar: float
    include_accel_loss: bool = True
    eta_f: float = 0.0
    use_spi_compress: bool = False
    spi_n: Optional[float] = None


# ================== WORKER TOP-LEVEL (usato dal ProcessPool) ==================
def process_one_p1(p1b: float, cfg: RunConfig) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if cfg.injector == "plain":
        res = run_plain_orifice_case(
            cfg.fluid, p1b, cfg.p2_bar, cfg.T_line,
            cfg.D, cfg.L, cfg.Cd, cfg.K_minor, cfg.rough,
            n_holes=cfg.n_holes, keep_total_area=cfg.keep_total_area,
            include_accel_loss=cfg.include_accel_loss, eta_f=cfg.eta_f,
            use_spi_compress=cfg.use_spi_compress, spi_n=cfg.spi_n
        )
        return res, None
    else:
        res = postprocess_case(
            fluid=cfg.fluid,
            p1_bar=p1b,
            p2_bar=cfg.p2_bar,
            T_line=cfg.T_line,
            D=cfg.D,
            L=cfg.L,
            Cd=cfg.Cd,
            K_minor=cfg.K_minor,
            rough=cfg.rough,
            include_accel_loss=cfg.include_accel_loss,
            eta_f=cfg.eta_f,
            use_spi_compress=cfg.use_spi_compress,
            spi_n=cfg.spi_n
        )
        spray = estimate_pressure_swirl_params(
            fluid=cfg.fluid,
            p1=p1b * 1e5, p2=cfg.p2_bar * 1e5, T_line=cfg.T_line,
            d_orif=cfg.D, mdot_1D=res["mdot_1D"],
            aircore_factor=cfg.swirl_aircore,
            theta_default_deg=cfg.swirl_theta,
            Csmd=cfg.swirl_Csmd, q_rr=cfg.swirl_qrr,
            swirl_profile="solid"   # <-- opzionale
        )

        spray.update(dict(p1_bar=p1b, D=cfg.D))
        return res, spray


# ================== CLI / MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="1D injector model for N2O (CFD-ready tables).")
    # Sweep / case singolo
    parser.add_argument("--p1-start", type=float, help="P1 start [bar]")
    parser.add_argument("--p1-stop",  type=float, help="P1 stop  [bar]")
    parser.add_argument("--p1-step",  type=float, default=1.0, help="P1 step  [bar]")
    parser.add_argument("--p1",       type=float, help="Single P1 [bar] (disables plotting)")
    parser.add_argument("--no-plot",  action="store_true", help="Disable plot even if sweeping")
    # Scelta iniettore
    parser.add_argument("--injector", choices=["plain","swirl"], default="plain",
                        help="Injector type: plain (orifice) or swirl (pressure-swirl empirical add-on).")
    # Plain orifice: multi-foro
    parser.add_argument("--n-holes", type=int, default=1, help="Number of holes for plain orifice.")
    parser.add_argument("--keep-total-area", action="store_true", default=True,
                        help="Interpret D as total equivalent diameter; split area across n holes.")
    # Swirler knobs
    # Swirler knobs
    parser.add_argument("--swirl-aircore", type=float, default=0.45, help="Air-core area factor φ (0–1).")
    parser.add_argument("--swirl-theta",   type=float, default=55.0, help="Half-cone angle [deg].")
    parser.add_argument("--swirl-Csmd",    type=float, default=2.25, help="SMD constant (tune 1.8–3.0).")
    parser.add_argument("--swirl-qrr",     type=float, default=3.5,  help="Rosin–Rammler q.")

    # NEW: perdite di accelerazione e fattore extra di perdite locali
    parser.add_argument("--no-accel-loss", action="store_true",
                        help="Disabilita la perdita di accelerazione all'uscita.")
    parser.add_argument("--eta-f", type=float, default=0.0,
                        help="Fattore additivo (senza unità) sulle perdite locali K (default 0).")

    # NEW: correzione di comprimibilità nello SPI
    parser.add_argument("--spi-compress", action="store_true",
                        help="Attiva correzione di comprimibilità nello SPI (Y').")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Esponente isentropico n per correzione SPI (se non dato → stima via K=ρa^2).")


    # === INPUT DI BASE (modifica qui per i tuoi default) ===
    fluid  = "NitrousOxide"
    T_line = 300.0      # K
    D      = 2e-3       # m (equivalente totale se --keep-total-area)
    L      = 10e-3      # m
    Cd     = 0.60
    K_minor= 1.0 / (Cd**2)
    rough  = 1e-5
    p2_bar = 43.0

    args = parser.parse_args()
    include_accel_loss = (not args.no_accel_loss)
    eta_f = float(args.eta_f)
    use_spi_compress = bool(args.spi_compress)
    spi_n = args.spi_n


    # Lista P1
    if args.p1 is not None:
        p1_list_bar = [float(args.p1)]
    elif args.p1_start is not None and args.p1_stop is not None:
        p1_list_bar = list(np.arange(float(args.p1_start), float(args.p1_stop) + 1e-9, float(args.p1_step)))
    else:
        p1_list_bar = list(np.arange(50.0, 70.0 + 1e-9, 1.0))  # default sweep

    # Tabella input
    inputs = dict(fluid=fluid, T_line=T_line, D=D, L=L, Cd=Cd, rough=rough, p2_bar=p2_bar,
                  injector=args.injector, n_holes=args.n_holes, keep_total_area=args.keep_total_area)
    print_inputs_table_en(inputs)

    # Config picklable per i worker
    cfg = RunConfig(
        injector=args.injector,
        n_holes=args.n_holes,
        keep_total_area=args.keep_total_area,
        swirl_aircore=args.swirl_aircore,
        swirl_theta=args.swirl_theta,
        swirl_Csmd=args.swirl_Csmd,
        swirl_qrr=args.swirl_qrr,
        fluid=fluid, T_line=T_line, D=D, L=L, Cd=Cd, K_minor=K_minor, rough=rough, p2_bar=p2_bar,
        include_accel_loss=include_accel_loss, eta_f=eta_f,
        use_spi_compress=use_spi_compress, spi_n=spi_n
    )


    results: list[dict] = []
    swirl_rows: list[dict] = []

    # === single o sweep ===
    if len(p1_list_bar) == 1:
        p1b = p1_list_bar[0]
        res, spray = process_one_p1(p1b, cfg)
        results.append(res)
        print_all_tables_en(results)
        print_single_result_en(res)
        if spray is not None:
            print_swirler_table_en([spray])
    else:
        max_workers = min(24, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut = {ex.submit(process_one_p1, p1b, cfg): p1b for p1b in p1_list_bar}
            for f in as_completed(fut):
                r, s = f.result()
                results.append(r)
                if s is not None:
                    swirl_rows.append(s)

        results.sort(key=lambda r: r["p1_bar"])
        print_all_tables_en(results)
        if cfg.injector == "swirl" and swirl_rows:
            swirl_rows.sort(key=lambda s: s["p1_bar"])
            print_swirler_table_en(swirl_rows)

        if (not args.no_plot) and len(results) >= 2:
            p1 = [r["p1_bar"] for r in results]
            plt.figure(figsize=(10, 6))
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
