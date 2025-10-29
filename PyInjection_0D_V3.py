"""
PyInjection_1D.py — Injector mass-flow model for N2O (NHNE baseline).
- Portata nominale: mdot_NHNE (blend SPI↔HEM)
- SPI/HEM sono calcolati per confronto/bounding
- Le proprietà di uscita (U_out, Re, Mach, rho_mix, ecc.) sono rese coerenti
  con mdot_NHNE tramite una piccola iterazione energetica.
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
MU_GAS_FALLBACK: float   = 1.85e-5   # Pa·s
MU_LIQ_FALLBACK: float   = 3.00e-4   # Pa·s
AOUT_GAS_FALLBACK: float = 203.44    # m/s (fallback a_out)

# Tuning opzionale per la dipendenza del blend NHNE dal residence time (L/D)
K_RESIDENCE_COEFF: float = 0       # 0=off; tipico 0.2–1.0

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
    """
    if D <= 0.0 or Cd <= 0.0 or p1 <= 0.0 or p2 <= 0.0 or p1 <= p2:
        return 0.0

    A  = 0.25 * math.pi * D * D
    dp = p1 - p2

    # densità di riferimento *liquida* (limite senza flashing)
    try:
        rho_ref = cp.PropsSI('D', 'P', p1, 'Q', 0, fluid)
    except Exception:
        rho_ref = rho_singlephase_at_T(fluid, p1, T_line, side="liq")

    mdot_ideal = Cd * A * math.sqrt(max(2.0 * rho_ref * dp, 0.0))
    if not use_compress:
        return mdot_ideal

    # ---- correzione di comprimibilità Y' ----
    if n_isentropic is not None and n_isentropic > 1.0:
        pr = max(p2 / p1, 1e-9)
        Yp = (n_isentropic / (n_isentropic - 1.0)) * (1.0 - pr**((n_isentropic - 1.0) / n_isentropic))
    else:
        # usa _safe_speed_of_sound per a1 (monofase liquida a monte)
        a1 = _safe_speed_of_sound(fluid, p1, T=T_line, two_phase=False)
        K  = max(rho_ref * a1 * a1, 1e5)   # guard-rail
        Yp = dp / K

    corr = 1.0 / math.sqrt(max(1.0 + Yp, 1e-6))
    return mdot_ideal * corr

def solve_mdot_hem(fluid: str, p1: float, p2: float, T_line: float,
                   D: float, Cd: float,
                   U_in: float = 0.0,
                   max_iter: int = 10000,
                   U_out_guess: Optional[float] = None) -> float:
    """
    HEM energetico minimale (micro-iterazione su h2 e rho_mix a p2).
    """
    if D <= 0.0 or Cd <= 0.0 or p1 <= 0.0 or p2 <= 0.0 or p1 <= p2:
        return 0.0

    A = 0.25 * math.pi * D * D
    try:
        h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    except Exception:
        return 0.0

    # init coerente a p2 con lato scelto in base a T_line vs T_sat(p2)
    try:
        p_crit = cp.PropsSI('Pcrit', fluid)
    except Exception:
        p_crit = float('inf')

    try:
        if p2 < p_crit:
            T_sat2 = cp.PropsSI('T', 'P', p2, 'Q', 0, fluid)  # = Q=1 stessa T
            side = "gas" if (T_line > T_sat2 + 0.5) else "liq"
        else:
            # sopra il critico: non esistono Q; scegli "liq" come start più denso e stabile
            side = "liq"

        rho0 = rho_singlephase_at_T(fluid, p2, T_line, side=side)
    except Exception:
        # fallback ultra-robusto: resta su p2
        rho0 = max(1.0, rho_singlephase_at_T(fluid, p2, T_line, side="liq"))


    mdot = Cd * A * math.sqrt(max(2.0 * rho0 * (p1 - p2), 0.0))

    for _ in range(max_iter):
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
    NHNE: blend continuo tra SPI e HEM
      mdot = (k/(k+1))·mdot_SPI + (1/(k+1))·mdot_HEM
    con k = sqrt((P1-P2) / max(Pv(T_line)-P2, eps)), opzionalmente modulato da L/D.
    """
    k = _k_classic_local(fluid, p_local=p1, p2=p2, T_line=T_line, L_over_D=L_over_D)

    mdot_spi = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd,
                              phase_out="auto", T_out=None, U_in=U_in, U_out_guess=None,
                              use_compress=use_spi_compress, n_isentropic=spi_n)
    mdot_hem = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd,
                              U_in=U_in, U_out_guess=None)

    mdot_nhne = (k/(k+1.0))*mdot_spi + (1.0/(k+1.0))*mdot_hem
    return mdot_nhne, k

# ================== stato uscita coerente con mdot_NHNE ==================
def nhne_out_state_from_mdot(fluid: str, p1: float, p2: float, T_line: float,
                             D: float, mdot_nhne: float, h1_hint: Optional[float] = None,
                             spi_phase_mode: str = "auto", max_iter: int = 10000) -> Dict[str, Any]:
    """
    Dato mdot (NHNE), rende consistenti le grandezze di uscita (U_out, rho_mix, T_out, x/alpha, mu_mix, a_out, Re, Mach).
    Itera su h2 tramite bilancio energetico con U_out = mdot / (rho_mix*A).
    """
    A   = 0.25 * math.pi * D**2
    mdot = max(mdot_nhne, 0.0)

    # h1 a monte (se disponibile passala, altrimenti calcolo da T_line)
    if h1_hint is not None:
        h1 = float(h1_hint)
    else:
        try:
            h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
        except Exception:
            h1 = cp.PropsSI('H', 'P', p1, 'Q', 0, fluid)

    # init: densità "liquida di riferimento" a p2
    try:
        rho2 = cp.PropsSI('D', 'P', p2, 'Q', 0, fluid)
    except Exception:
        T_sat_p2 = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
        rho2 = rho_singlephase_at_T(fluid, p2, max(T_sat_p2 - 0.5, 100.0), side="liq")

    T_out, x_out, alpha_out = T_line, 0.0, 0.0
    rho_l, rho_v = None, None
    is_two = False

    for _ in range(max_iter):
        U_out = mdot / max(rho2 * A, 1e-12)
        T_out, h2, phase_hint = estimate_T_out_energy(fluid, p1, T_line, p2, U_out=U_out, U_in=0.0)

        # Stato di miscela a (p2, h2) → x, rho_mix
        rho_mix, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p2, h2)
        if rho_mix is None:
            # monofase: usa SPI auto per lato
            try:
                T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
            except Exception:
                T_sat = T_out
            if (T_out > T_sat + 0.5):
                rho_mix = rho_singlephase_at_T(fluid, p2, T_out, side="gas")
                x, is_two = 1.0, False
            else:
                rho_mix = rho_singlephase_at_T(fluid, p2, T_out, side="liq")
                x, is_two = 0.0, False

        # controllo convergenza su rho2
        if abs(rho_mix - rho2) <= 1e-3 * max(rho2, 1.0):
            rho2 = rho_mix
            break
        rho2 = rho_mix

    # viscosità “mix” per Re
    if is_two:
        mu_l = _safe_viscosity(fluid, p2, phase="liq")
        mu_g = _safe_viscosity(fluid, p2, T_out, phase="gas")
        mu_mix = (1.0 - x) * mu_l + x * mu_g
    else:
        try:
            T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
        except Exception:
            T_sat = T_out
        mu_mix = _safe_viscosity(fluid, p2, T_out, phase=("gas" if T_out > T_sat + 0.5 else "liq"))

    # frazione volumetrica alpha
    if is_two and (rho_l and rho_v):
        Vv = x / max(rho_v, 1e-12)
        Vl = (1.0 - x) / max(rho_l, 1e-12)
        alpha_out = Vv / max(Vv + Vl, 1e-12)
    else:
        alpha_out = float(x >= 0.999)

    # velocità e numeri adimensionali
    U_out = mdot / max(rho2 * A, 1e-12)
    try:
        T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
    except Exception:
        T_sat = T_out
    a_out = _safe_speed_of_sound(fluid, p2, T_out, two_phase=is_two, x=x, rho_l=rho_l, rho_v=rho_v)
    Mach  = U_out / max(a_out, 1e-9)
    Re    = rho2 * U_out * D / max(mu_mix, 1e-12)

    # portate/velocità di fase (diagnostiche, coerenti con lo stato)
    Vdot = mdot / max(rho2, 1e-12)
    if is_two:
        Vdot_v = x * mdot / max(rho_v, 1e-12)
        Vdot_l = (1.0 - x) * mdot / max(rho_l, 1e-12)
    else:
        if x >= 0.999:     # gas monofase
            Vdot_v = Vdot
            Vdot_l = 0.0
        elif x <= 1e-12:   # liquido monofase
            Vdot_l = Vdot
            Vdot_v = 0.0
        else:
            Vdot_l = (1.0 - x) * Vdot
            Vdot_v = x * Vdot

    j_liq = Vdot_l / A
    j_gas = Vdot_v / A


    # --- Densità di fase coerenti con lo stato NHNE (per stampa) ---
    # BIFASE: usa le densità di fase da HEM (alla saturazione a p2).
    # MONOFASE: usa lo stato reale a (p2, T_out) sul lato corretto.
    if is_two:
        rho_l_nhne = rho_l  # già da mixture_rho_HEM
        rho_v_nhne = rho_v
    else:
        # determina lato in base a T_out vs T_sat(p2)
        try:
            T_sat = cp.PropsSI('T', 'P', p2, 'Q', 1, fluid)
        except Exception:
            T_sat = T_out
        if T_out > T_sat + 0.5:
            # gas monofase reale
            rho_v_nhne = rho_singlephase_at_T(fluid, p2, T_out, side="gas")
            rho_l_nhne = None
        else:
            # liquido monofase reale
            rho_l_nhne = rho_singlephase_at_T(fluid, p2, T_out, side="liq")
            rho_v_nhne = None


    return dict(
        T_out=T_out, rho_mix=rho2, mu_mix=mu_mix, a_out=a_out,
        U_out=U_out, Mach=Mach, Re_out=Re,
        x_out=float(x), alpha_out=float(min(max(alpha_out, 0.0), 1.0)),
        Vdot_l=Vdot_l, Vdot_v=Vdot_v, j_liq=j_liq, j_gas=j_gas,
        rho_l=rho_l_nhne, rho_v=rho_v_nhne,
        phase_out=("two-phase" if is_two else ("gas" if T_out > T_sat + 0.5 else "liquid")),
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
    for k in ["mdot_spi", "mdot_hem", "mdot_nhne", "Vdot_l", "Vdot_v", "mdot_liq", "mdot_gas"]:
        res[k] *= n
    res["A"] *= n
    res.update(dict(n_holes=n, D_per_hole=D_hole, A_total=res["A"], A_per_hole=res["A"]/n))
    return res

# ================== PRESSURE-SWIRL EMPIRICAL PACK ==================
def estimate_pressure_swirl_params(fluid: str,
                                   p1: float, p2: float, T_line: float, d_orif: float,
                                   mdot: float,
                                   aircore_factor: float = 0.45,
                                   theta_default_deg: float = 55.0,
                                   Csmd: float = 2.25,
                                   dist_type: str = "Rosin-Rammler",
                                   q_rr: float = 3.5,
                                   swirl_profile: str = "free") -> Dict[str, Any]:
    """Stime empiriche per pressure-swirl: (θ, A_eff, U0, SMD, distribuzione, swirl number)."""
    d = float(d_orif)
    A_orif = 0.25 * math.pi * d**2
    phi = min(max(float(aircore_factor), 0.0), 0.99)  # frazione di area liquida
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
        sigma = cp.PropsSI('I', 'P', p2, 'Q', 0, fluid) # tensione superficiale
    except Exception:
        sigma = 0.010  # N/m

    # idraulica base + SMD Lefebvre-like
    dP = max(p1 - p2, 0.0)
    U0 = mdot / max(rho_l * A_eff, 1e-12)   # velocità media assiale nell’anello
    theta_deg = float(theta_default_deg)

    SMD = Csmd * ((mu_l**0.25) * (sigma**0.25)) / ((rho_l**0.25) * (max(dP, 1.0)**0.5))
    SMD *= max(d, 1e-6)**0.25
    SMD = min(max(SMD, 5e-6), 300e-6)

    # === Swirl ratio & Swirl number ===
    theta_rad = math.radians(theta_deg)
    swirl_ratio = math.tan(theta_rad)  # Ut/Ux

    prof = (swirl_profile or "free").lower()
    if prof not in ("free", "solid"):
        prof = "free"

    if prof == "free":
        swirl_number = swirl_ratio                     # S = tan θ (vortice libero + Ux uniforme)
    else:
        swirl_number = (2.0 / 3.0) * swirl_ratio       # S = (2/3) tan θ (rotazione solida)

    return dict(
        theta_deg=theta_deg,
        A_eff=A_eff,
        U0=U0,
        SMD=SMD,
        d32=SMD,
        dist=dist_type,
        q_rr=q_rr,
        mdot=mdot,
        swirl_ratio=swirl_ratio,
        swirl_number=swirl_number,
        swirl_profile=prof,
        notes="Tune aircore_factor, theta_deg, Csmd se disponi di dati specifici. S calcolato da θ."
    )

# ================== POST-PROCESS (NHNE baseline) ==================
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

    # entalpia a monte (hint)
    try:
        h1 = cp.PropsSI('H', 'P', p1, 'T', T_line, fluid)
    except Exception:
        h1 = cp.PropsSI('H', 'P', p1, 'Q', 0, fluid)

    # 0D (confronto)
    mdot_spi  = solve_mdot_spi(fluid, p1, p2, T_line, D, Cd,
                               phase_out="auto", T_out=None, U_in=0.0, U_out_guess=None,
                               use_compress=use_spi_compress, n_isentropic=spi_n)
    mdot_hem  = solve_mdot_hem(fluid, p1, p2, T_line, D, Cd, U_in=0.0, U_out_guess=None)
    mdot_nhne, kv = solve_mdot_nhne(fluid, p1, p2, T_line, D, Cd,
                                    U_in=0.0, U_out_guess=None,
                                    use_spi_compress=use_spi_compress, spi_n=spi_n,
                                    L_over_D=(L / max(D, 1e-12)))

    # Stato d’uscita coerente con mdot_NHNE
    out = nhne_out_state_from_mdot(fluid, p1, p2, T_line, D, mdot_nhne, h1_hint=h1)

    # diagnosi ΔT
    try:
        T_sat = cp.PropsSI('T','P',p2,'Q',1,fluid)
    except Exception:
        T_sat = out["T_out"]
    dT = out["T_out"] - T_sat

        # scelta modello CFD (monofase → LIQ/GAS, due-fasi → DPM/VOF)
    if out["phase_out"] == "two-phase":
        a = out["alpha_out"]
        if   a < 0.20:  CFD_model = "DPM"
        elif a <= 0.80: CFD_model = "VOF"
        else:           CFD_model = "GAS+VOF(inj)"
    else:
        # monofase
        CFD_model = "LIQ" if out["phase_out"] == "liquid" else "GAS"


    # warnings
    warnings = []
    if out["Mach"] > 0.9: warnings.append(f"High Mach at outlet (M={out['Mach']:.2f}).")
    try:
        Tc = cp.PropsSI('Tcrit', fluid); Pc = cp.PropsSI('pcrit', fluid)
        if (abs(out["T_out"] - Tc) < 2.0) or (abs(p2 - Pc)/Pc < 0.03):
            warnings.append("Near-critical region: properties may be stiff; treat with caution.")
    except Exception:
        pass
    if iso_ratio == iso_ratio and iso_ratio <= 0.80:
        warnings.append(f"Isolating regime likely (P2/Psat(T_line) = {iso_ratio:.2f}).")


    return dict(
        p1_bar=p1_bar, A=A,
        # portate (baseline = NHNE)
        mdot_nhne=mdot_nhne, mdot_spi=mdot_spi, mdot_hem=mdot_hem,
        k_nhne=kv,
        # stato uscita coerente con mdot_nhne
        U_out=out["U_out"],
        U_mix=(mdot_nhne / max(out["rho_mix"], 1e-12) / max(A, 1e-20)),
        T_out=out["T_out"], T_sat=T_sat, dT=dT,
        rho_mix=out["rho_mix"], mu_mix=out["mu_mix"], a_out=out["a_out"],
        Mach=out["Mach"], Re_out=out["Re_out"],
        x_out=out["x_out"], alpha_out=out["alpha_out"],
        rho_l=out["rho_l"], rho_v=out["rho_v"],
        Vdot_l=out["Vdot_l"], Vdot_v=out["Vdot_v"],
        mdot_liq=(1.0 - out["x_out"]) * mdot_nhne, mdot_gas=out["x_out"] * mdot_nhne,
        j_liq=out["j_liq"], j_gas=out["j_gas"],
        phase_out=out["phase_out"], CFD_model=CFD_model,
        iso_ratio=iso_ratio, warnings=warnings
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
        ("P1 [bar]",          "p1_bar",     8, ".2f"),
        ("mdot_SPI [kg/s]",   "mdot_spi",  15, ".5f"),
        ("mdot_HEM [kg/s]",   "mdot_hem",  15, ".5f"),
        ("mdot_NHNE [kg/s]",  "mdot_nhne", 16, ".5f"),
        ("k_NHNE [-]",        "k_nhne",    12, ".3f"),
        ("x_out [-]",         "x_out",     10, ".4f"),
        ("alpha_out [-]",     "alpha_out", 13, ".4f"),
        ("Vdot_l [m^3/s]",    "Vdot_l",    16, ".5f"),
        ("Vdot_v [m^3/s]",    "Vdot_v",    16, ".5f"),
    ]
    print_table_en(
        "\nRESULT TABLE – Mass flow (NHNE baseline) and phase indicators",
        cols_results, results
    )

    cols_cfd = [
        ("P1 [bar]",         "p1_bar",     8,  ".2f"),
        ("T_out [K]",        "T_out",     10,  ".2f"),
        ("T_sat(P2) [K]",    "T_sat",     13,  ".2f"),
        ("ΔT [K]",           "dT",         8,  ".2f"),
        ("P2/Psat(T1) [-]",  "iso_ratio", 14,  ".2f"),
        ("rho_mix [kg/m^3]", "rho_mix",   18,  ".3f"),
        ("mu_mix [Pa·s]",    "mu_mix",    16,  ".3e"),
        ("a_out [m/s]",      "a_out",     12,  ".2f"),
        ("Mach [-]",         "Mach",      10,  ".3f"),
        ("Re [-]",           "Re_out",    13,  ".2e"),
        ("rho_l [kg/m^3]",   "rho_l",     16,  ".3f"),
        ("rho_v [kg/m^3]",   "rho_v",     16,  ".5f"),
    ]
    print_table_en(
        "CFD-READY TABLE – Outlet properties consistent with mdot_NHNE",
        cols_cfd, results
    )

    cols_phases = [
        ("P1 [bar]",        "p1_bar",     8,  ".2f"),
        ("phase_out",       "phase_out", 12,  "s"),
        ("CFD_model",       "CFD_model", 12,  "s"),
        ("mdot_liq [kg/s]", "mdot_liq",  16,  ".5f"),
        ("mdot_gas [kg/s]", "mdot_gas",  16,  ".5f"),
        ("U_out [m/s]",     "U_out",     12,  ".2f"),
        ("U_mix [m/s]",     "U_mix",     12,  ".2f"),
        ("j_liq [m/s]",     "j_liq",     12,  ".2f"),
        ("j_gas [m/s]",     "j_gas",     12,  ".2f"),
    ]
    print_table_en(
        "PHASE TABLE (for CFD setup) – Phase mass flows and velocities",
        cols_phases, results
    )

    print("LEGEND:")
    print(" - mdot_NHNE è la portata nominale (blend SPI↔HEM).")
    print(" - U_out è coerente con mdot_NHNE e le proprietà di uscita.")
    print(" - x_out: qualità in massa; alpha_out: frazione volumetrica di vapore.\n")

def print_single_result_en(res: dict) -> None:
    print(f"\n=== SINGLE CASE RESULT — P1 = {res['p1_bar']:.2f} bar ===")
    print(f"mdot_NHNE = {res['mdot_nhne']:.6f} kg/s | U_out = {res['U_out']:.2f} m/s "
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
            d_orif=cfg.D, mdot_1D=res["mdot_nhne"],  # usa NHNE come mdot di riferimento
            aircore_factor=cfg.swirl_aircore,
            theta_default_deg=cfg.swirl_theta,
            Csmd=cfg.swirl_Csmd, q_rr=cfg.swirl_qrr,
            swirl_profile="solid"
        )
        spray.update(dict(p1_bar=p1b, D=cfg.D))
        return res, spray

# ================== CLI / MAIN ==================
def main():
    parser = argparse.ArgumentParser(description="Injector model for N2O (NHNE baseline, CFD-ready tables).")
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
    parser.add_argument("--swirl-aircore", type=float, default=0.45, help="Air-core area factor φ (0–1).")
    parser.add_argument("--swirl-theta",   type=float, default=55.0, help="Half-cone angle [deg].")
    parser.add_argument("--swirl-Csmd",    type=float, default=2.25, help="SMD constant (tune 1.8–3.0).")
    parser.add_argument("--swirl-qrr",     type=float, default=3.5,  help="Rosin–Rammler q.")

    # NEW: correzione di comprimibilità nello SPI
    parser.add_argument("--spi-compress", action="store_true",
                        help="Attiva correzione di comprimibilità nello SPI (Y').")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Esponente isentropico n per correzione SPI (se non dato → stima via K=ρa^2).")

    # === INPUT DI BASE (modifica qui per i tuoi default) ===
    fluid  = "NitrousOxide"
    T_line = 288.0      # K
    D      = 2e-3       # m (equivalente totale se --keep-total-area)
    L      = 10e-3      # m
    Cd     = 0.9875
    K_minor= 1.0 / (Cd**2)
    rough  = 1e-5
    p2_bar = 43.0

    args = parser.parse_args()
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
        include_accel_loss=True, eta_f=0.0,
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
            plt.plot(p1, [r["mdot_spi"]   for r in results], 's--', label='mdot_SPI (single-phase)', linewidth=2)
            plt.plot(p1, [r["mdot_hem"]   for r in results], 'd--', label='mdot_HEM (equilibrium two-phase)', linewidth=2)
            plt.plot(p1, [r["mdot_nhne"]  for r in results], 'o-',  label='mdot_NHNE (baseline)', linewidth=2)
            plt.xlabel('Inlet Pressure $P_1$ [bar]')
            plt.ylabel(r'Mass Flow Rate $\dot{m}$ [kg/s]')
            plt.title('Mass Flow vs Inlet Pressure')
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
