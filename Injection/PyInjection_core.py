"""
pyinjection_core.py
-------------------

Core backend for the plain-orifice N2O injector:

- Mass-flow models:
    * SPI  = single-phase incompressible/compressible
    * HEM  = homogeneous equilibrium model
    * NHNE = non-homogeneous non-equilibrium blend of SPI/HEM
- Phase-aware selector (SPI vs NHNE)
- Outlet thermodynamic state consistent with mdot
  (HEM + stagnation enthalpy balance)
- Robust thermodynamic property helpers (especially close to saturation)
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any

import CoolProp.CoolProp as cp

# =============================================================================
# Experimental sources used to calibrate Cd(r/D, L/D, Re) for short orifices
# =============================================================================
#
# The discharge-coefficient model implemented in this module is NOT a direct
# fit of a single paper. It is a simplified and regularized synthesis of trends
# observed in several experimental datasets. It is intended as a first-level
# estimate consistent with the literature, not as a substitute for detailed
# testing in the final design phase.
#
# Main experimental references:
#
# [1] Reader–Harris, M.J., Gallagher, P.M. (1998).
#     "The Orifice Plate Discharge Coefficient Equation".
#     Basis of ISO 5167 for Cd(Re, β) in sharp-edged orifice plates.
#
# [2] Gelalles, A.G., Marsh, E.T. (1931).
#     "Effect of Orifice Length-Diameter Ratio on the Coefficient of Discharge".
#     Historical data on L/D effects in short cylindrical orifices.
#
# [3] Edlebeck, S. (2013).
#     "Measurements of the Flow of Supercritical CO₂ Through Short Orifices".
#     Influence of larger L/D and trans/supercritical regimes.
#
# [4] Waxman, J., Dyer, J., Karabeyoglu, A. (2019, Stanford).
#     Measurements of Cd for single-orifice N₂O injectors with
#     sharp-edged / chamfered / rounded inlets.
#
# [5] Internal dataset provided by the user:
#     "Discharge Coefficient (Cd) in Short Orifices for N₂O (and Similar Fluids)
#      – Dataset and Correlations.pdf".
#     → Used to align the numerical ranges with typical N₂O conditions
#       in the field 10⁴ ≲ Re ≲ 10⁵ and 0.5 ≲ L/D ≲ 10.
#
# =============================================================================


# ----------------------------------------------------------------------
# Guard parameters
# ----------------------------------------------------------------------
DELTA_T_HYST = 0.5          # [K] hysteresis for comparisons vs Tsat
EPS_REL_PSAT = 1e-4         # relative tolerance on |P - Psat|/Psat
P_EPS        = 5.0e4        # [Pa] epsilon for pressure denominators
PKEY_DPA     = 1.0e2        # [Pa] pressure quantization step

# Physical fallbacks for properties
MU_GAS_FALLBACK: float   = 1.85e-5   # Pa·s
MU_LIQ_FALLBACK: float   = 3.00e-4   # Pa·s
AOUT_GAS_FALLBACK: float = 203.44    # m/s (fallback a_out, gas side)


# ----------------------------------------------------------------------
# Robust saturation utilities
# ----------------------------------------------------------------------
def _pkey(p: float) -> float:
    """Quantize pressure to stabilize property calls near saturation."""
    return float(round(p / PKEY_DPA) * PKEY_DPA)


def _safe_psat_at_T(fluid: str, T: float) -> float:
    """
    Robust Psat(T). If not defined (above Tc or error), fall back to pcrit
    or 1 bar.
    """
    try:
        return cp.PropsSI("P", "T", T, "Q", 1, fluid)
    except Exception:
        try:
            return cp.PropsSI("pcrit", fluid)
        except Exception:
            return 1.0e5  # extreme fallback: 1 bar


def _safe_tcrit_pcrit(fluid: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (Tcrit, pcrit) safely, or (None, None) on error."""
    try:
        Tc = cp.PropsSI("Tcrit", fluid)
    except Exception:
        Tc = None
    try:
        Pc = cp.PropsSI("pcrit", fluid)
    except Exception:
        Pc = None
    return Tc, Pc


def _safe_tsat_at_p(fluid: str, p: float) -> Optional[float]:
    """
    Robust Tsat(P): temperature such that Q = 1 at pressure P;
    if supercritical or on error → None.
    """
    Tc, Pc = _safe_tcrit_pcrit(fluid)
    if (Pc is not None) and (p >= Pc):
        return None
    try:
        return cp.PropsSI("T", "P", _pkey(p), "Q", 1, fluid)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Single-phase density near saturation
# ----------------------------------------------------------------------
def _rho_single_at_T(fluid: str, p: float, T: float, side: str) -> float:
    """
    Robust single-phase density near saturation.

    If |p - Psat(T)|/Psat(T) < EPS_REL_PSAT, shift p on the chosen side:
      - side = 'gas' → slightly below Psat
      - side = 'liq' → slightly above Psat
    """
    try:
        p_sat = _safe_psat_at_T(fluid, T)
        if abs(p - p_sat) / max(p_sat, 1.0) < EPS_REL_PSAT:
            p = (0.999 * p_sat) if (side == "gas") else (1.001 * p_sat)
    except Exception:
        pass
    return cp.PropsSI("D", "P", _pkey(p), "T", T, fluid)


# ----------------------------------------------------------------------
# Local blend factor k for NHNE
# ----------------------------------------------------------------------
def _k_local(
    fluid: str,
    p_local: float,
    p2: float,
    T_line: float,
) -> float:
    """
    Local blending factor k for NHNE:

        k = sqrt( (p_local - p2) / max(Psat(T_line) - p2, P_EPS) )

    Slightly filtered if p_local < p2 or beyond saturation,
    so that k ≥ 0 and continuous.

    Note:
    Any dependence on L/D (residence-time-like corrections) is intentionally
    not included here, because geometric effects are already captured in the
    discharge coefficient Cd.
    """
    pV = _safe_psat_at_T(fluid, T_line)
    den = max(pV - p2, P_EPS)
    num = max(p_local - p2, 0.0)
    k = math.sqrt(num / den)
    return k

# ======================================================================
#                            MASS-FLOW MODELS
# ======================================================================
def _mdot_spi(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    D: float,
    Cd: float,
    use_compress: bool = True,
    n_isentropic: Optional[float] = None,
) -> float:
    """
    SPI mass-flow model:

      - Select upstream phase (gas vs liquid) from T_line vs Tsat(P1)
      - Optional compressibility correction Y'

    Returns mdot_SPI [kg/s].
    """
    if D <= 0.0 or Cd <= 0.0 or p1_bar <= p2_bar:
        return 0.0

    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)
    A = 0.25 * math.pi * D * D
    dp = p1 - p2

    # Upstream phase
    Tc, _ = _safe_tcrit_pcrit(fluid)
    try:
        T_sat1 = cp.PropsSI("T", "P", p1, "Q", 1, fluid)
        upstream_is_gas = (T_line > T_sat1 + DELTA_T_HYST)
    except Exception:
        upstream_is_gas = (Tc is not None) and (T_line >= Tc)

    rho_ref = _rho_single_at_T(
        fluid,
        p1,
        T_line,
        side=("gas" if upstream_is_gas else "liq"),
    )

    mdot_spi = Cd * A * math.sqrt(max(2.0 * rho_ref * dp, 0.0))
    if not use_compress:
        return mdot_spi

    # Compressibility correction
    if n_isentropic and (n_isentropic > 1.0):
        pr = max(p2 / p1, 1e-12)
        Yp = (n_isentropic / (n_isentropic - 1.0)) * (
            1.0 - pr ** ((n_isentropic - 1.0) / n_isentropic)
        )
    else:
        a1 = cp.PropsSI("A", "P", p1, "T", T_line, fluid)
        K = max(rho_ref * a1 * a1, 1e5)
        Yp = dp / K

    return mdot_spi / math.sqrt(max(1.0 + Yp, 1e-6))

def _mdot_hem(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    D: float,
    Cd: float,
) -> float:
    """
    Minimal energetic HEM model (mass-flow only) with safety checks:

      - h1 from (P1, T_line) or from saturated liquid (Q=0)
      - iteration on h2 and rho(P2, H2)

    In addition, we impose a physical upper bound:
    the HEM mass flow rate cannot exceed the incompressible
    single-phase SPI mass flow rate with the same Cd.
    """
    if D <= 0.0 or Cd <= 0.0 or p1_bar <= p2_bar:
        return 0.0

    # --- Physical upper bound: incompressible SPI with same Cd ---
    mdot_spi_upper = _mdot_spi(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        Cd=Cd,
        use_compress=False,   # incompressible reference
        n_isentropic=None,
    )

    p1 = _pkey(p1_bar * 1e5)
    p2 = _pkey(p2_bar * 1e5)
    A = 0.25 * math.pi * D * D

    # Upstream enthalpy
    try:
        h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
    except Exception:
        # fallback: saturated liquid
        h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    # Initial guess: liquid at outlet
    try:
        rho0 = cp.PropsSI("D", "P", p2, "Q", 0, fluid)
    except Exception:
        rho0 = _rho_single_at_T(fluid, p2, T_line, side="liq")

    mdot = Cd * A * math.sqrt(max(2.0 * rho0 * (p1 - p2), 0.0))

    for _ in range(1000):
        U = mdot / max(rho0 * A, 1e-12)
        h2 = h1 - 0.5 * U * U

        try:
            rho = cp.PropsSI("D", "P", p2, "H", h2, fluid)
        except Exception:
            rho = rho0

        try:
            h_f2 = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        except Exception:
            h_f2 = h2

        deltah = max(h2 - h_f2, 0.0)
        mdot_new = Cd * A * rho * math.sqrt(max(2.0 * deltah, 0.0))

        if abs(mdot_new - mdot) <= 1e-3 * max(mdot, 1.0):
            mdot = mdot_new
            break

        mdot, rho0 = mdot_new, rho

    # Final clamp: HEM cannot exceed incompressible SPI
    mdot = max(mdot, 0.0)
    if mdot_spi_upper > 0.0 and mdot > mdot_spi_upper:
        mdot = mdot_spi_upper

    return mdot

def _mdot_nhne(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    D: float,
    Cd: float,
    use_spi_compress: bool = True,
    spi_n: Optional[float] = None,
) -> Tuple[float, float]:
    """
    NHNE mass-flow model:

        mdot_NHNE = [k/(k+1)] * mdot_SPI + [1/(k+1)] * mdot_HEM

    Returns (mdot_nhne, k).

    Note:
    The local blending factor k depends only on pressure and saturation
    at T_line. Any residence-time-like dependence on L/D is NOT applied
    here, as geometric effects are already included in Cd.
    """
    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)

    mdot_spi = _mdot_spi(
        fluid,
        p1_bar,
        p2_bar,
        T_line,
        D,
        Cd,
        use_compress=use_spi_compress,
        n_isentropic=spi_n,
    )
    mdot_hem = _mdot_hem(fluid, p1_bar, p2_bar, T_line, D, Cd)

    k = _k_local(
        fluid,
        p_local=p1,
        p2=p2,
        T_line=T_line,
    )

    mdot_nhne = (k / (k + 1.0)) * mdot_spi + (1.0 / (k + 1.0)) * mdot_hem
    return mdot_nhne, k

# ----------------------------------------------------------------------
# Minimal classifier to decide if SPI produces two-phase outlet
# ----------------------------------------------------------------------
def _phase_from_spi_guess(
    fluid: str,
    p1: float,
    p2: float,
    T_line: float,
    D: float,
    mdot_spi: float,
) -> str:
    """
    Decide whether the outlet would be two-phase using the SPI mass flow.

    Uses stagnation-enthalpy balance plus a saturation band at P2.

    Returns: 'two-phase' | 'liquid' | 'gas'
    """
    A = 0.25 * math.pi * D * D

    try:
        h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
    except Exception:
        h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    try:
        rho2 = cp.PropsSI("D", "P", p2, "Q", 0, fluid)
    except Exception:
        rho2 = _rho_single_at_T(fluid, p2, T_line, side="liq")

    U = mdot_spi / max(rho2 * A, 1e-12)
    h2 = h1 - 0.5 * U * U

    T_sat2 = _safe_tsat_at_p(fluid, p2)
    have_sat = (T_sat2 is not None)

    if have_sat:
        try:
            h_f2 = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
            h_g2 = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
            if h_f2 <= h2 <= h_g2:
                return "two-phase"
        except Exception:
            pass

    # Monophasic: decide gas vs liquid
    try:
        T_out = cp.PropsSI("T", "P", p2, "H", h2, fluid)
    except Exception:
        T_out = T_line

    if have_sat:
        return "gas" if (T_out > T_sat2 + DELTA_T_HYST) else "liquid"
    else:
        return "gas" if (T_out >= T_line) else "liquid"


# ----------------------------------------------------------------------
# Phase-aware public API
# ----------------------------------------------------------------------
def compute_mdot_phaseaware(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    D: float,
    Cd: float,
    *,
    L: Optional[float] = None,
    use_spi_compress: bool = True,
    spi_n: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Phase-aware mass-flow API.

    Returns (mdot_phaseaware, model_used) with automatic selection:

      - If the SPI guess leads to a two-phase outlet → use NHNE
      - Otherwise → use SPI

    model_used is one of {"SPI", "NHNE"}.

    Note:
    The NHNE blending factor k no longer includes any L/D-based
    residence-time correction. L is kept only for possible future
    extensions, while geometric effects are included in Cd.
    """
    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)

    mdot_spi = _mdot_spi(
        fluid,
        p1_bar,
        p2_bar,
        T_line,
        D,
        Cd,
        use_compress=use_spi_compress,
        n_isentropic=spi_n,
    )
    mdot_nhne, _ = _mdot_nhne(
        fluid,
        p1_bar,
        p2_bar,
        T_line,
        D,
        Cd,
        use_spi_compress=use_spi_compress,
        spi_n=spi_n,
    )

    phase_spi = _phase_from_spi_guess(fluid, p1, p2, T_line, D, mdot_spi)

    if phase_spi == "two-phase":
        return mdot_nhne, "NHNE"
    else:
        return mdot_spi, "SPI"


# ======================================================================
#                           OUTLET PROPERTY BLOCK
# ======================================================================
def _safe_viscosity(
    fluid: str,
    p: float,
    T: Optional[float] = None,
    phase: str = "gas",
    x: Optional[float] = None,
) -> float:
    """
    Robust viscosity:

      - phase = 'gas' / 'liq' → single-phase viscosity
      - any other value → two-phase blend (HEM-style) based on x

    x is the mass quality (gas mass fraction).
    """
    try:
        if phase == "gas":
            if T is None:
                return MU_GAS_FALLBACK
            return cp.PropsSI("V", "P", p, "T", T, fluid)

        if phase == "liq":
            if T is not None:
                # Slightly shift pressure away from Psat(T) on the liquid side
                try:
                    p_sat = cp.PropsSI("P", "T", T, "Q", 1, fluid)
                    p_eff = max(p, 1.001 * p_sat)
                except Exception:
                    p_eff = p
                return cp.PropsSI("V", "P", p_eff, "T", T, fluid)
            return cp.PropsSI("V", "P", p, "Q", 0, fluid)

        # Two-phase mixture: HEM-like blend
        xx = 0.5 if (x is None) else float(min(max(x, 0.0), 1.0))
        mu_l = _safe_viscosity(fluid, p, T=None, phase="liq")
        mu_g = (
            _safe_viscosity(fluid, p, T=T, phase="gas")
            if T is not None
            else _safe_viscosity(fluid, p, phase="gas")
        )
        return (1.0 - xx) * mu_l + xx * mu_g

    except Exception:
        return MU_GAS_FALLBACK if phase == "gas" else MU_LIQ_FALLBACK


def _safe_speed_of_sound(
    fluid: str,
    p: float,
    T: Optional[float] = None,
    two_phase: bool = False,
    x: Optional[float] = None,
    rho_l: Optional[float] = None,
    rho_v: Optional[float] = None,
) -> float:
    """
    Robust speed of sound:

      - For two-phase conditions: approximate with gas-side value at Tsat(P)
      - Otherwise: a(P, T) single-phase

    Currently this is an intentionally simple model for two-phase mixtures.
    """
    try:
        if two_phase:
            # Approximate using gas-side sound speed at (P, Tsat)
            T_sat = cp.PropsSI("T", "P", p, "Q", 1, fluid)
            T_use = max(T or (T_sat + 0.5), T_sat + 0.5)
            return cp.PropsSI("A", "P", p, "T", T_use, fluid)

        if T is None:
            return AOUT_GAS_FALLBACK

        return cp.PropsSI("A", "P", p, "T", T, fluid)

    except Exception:
        return AOUT_GAS_FALLBACK


def rho_singlephase_at_T(
    fluid: str,
    p: float,
    T: float,
    side: str = "gas",
) -> float:
    """
    Public helper: robust density near saturation forcing 'gas' or 'liq'
    by slightly shifting P away from Psat(T).

    side = 'gas' → slightly below Psat
    side = 'liq' → slightly above Psat
    """
    try:
        p_sat = cp.PropsSI("P", "T", T, "Q", 1, fluid)
        if abs(p - p_sat) / max(p_sat, 1.0) < 1e-4:
            p_safe = (0.999 * p_sat) if (side == "gas") else (1.001 * p_sat)
        else:
            p_safe = p
    except Exception:
        p_safe = p

    return cp.PropsSI("D", "P", _pkey(p_safe), "T", T, fluid)


def mixture_rho_HEM(
    fluid: str,
    p: float,
    h: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], float, bool]:
    """
    HEM mixture density at (p, h).

    Returns:
      (rho_mix, rho_l, rho_v, x, is_two)

    where:
      - rho_mix: mixture density [kg/m³] or None if outside saturation region
      - rho_l  : saturated liquid density at p
      - rho_v  : saturated vapour density at p
      - x      : mass quality (gas mass fraction)
      - is_two : True if 0 ≤ x ≤ 1 and mixture is within saturation band
    """
    try:
        h_f = cp.PropsSI("H", "P", p, "Q", 0, fluid)
        h_g = cp.PropsSI("H", "P", p, "Q", 1, fluid)
        rho_l = cp.PropsSI("D", "P", p, "Q", 0, fluid)
        rho_v = cp.PropsSI("D", "P", p, "Q", 1, fluid)

        x = (h - h_f) / (h_g - h_f) if h_g > h_f else -1.0
        if -1e-8 <= x <= 1.0 + 1e-8:
            x = min(1.0, max(0.0, x))
    except Exception:
        x = -1.0
        rho_l = rho_v = None

    if 0.0 <= x <= 1.0 and rho_l is not None and rho_v is not None:
        rho_m = 1.0 / (x / rho_v + (1.0 - x) / rho_l)
        return rho_m, rho_l, rho_v, x, True

    return None, None, None, 0.0, False


def estimate_T_out_energy(
    fluid: str,
    p1: float,
    T_line: float,
    p2: float,
    U_out: float,
    U_in: float = 0.0,
) -> Tuple[float, float, str]:
    """
    Estimate outlet temperature T_out from stagnation-enthalpy balance:

        h2 = h1 + U_in^2/2 - U_out^2/2

    Returns:
      (T_out, h2, phase_hint)

    where phase_hint ∈ {'gas', 'liq', 'two_phase'}.
    """
    # Upstream enthalpy h1
    try:
        h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
    except Exception:
        h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    h2 = h1 + 0.5 * (U_in * U_in - U_out * U_out)
    h2 = max(h2, h1 - 1e7)  # crude lower bound

    # Saturation at outlet
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
        h_f2 = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        h_g2 = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = None
        h_f2 = h_g2 = None

    # Two-phase?
    if (h_f2 is not None) and (h_g2 is not None) and (h_f2 <= h2 <= h_g2):
        return (T_sat if T_sat is not None else T_line), h2, "two_phase"

    # Single-phase
    try:
        T_out = cp.PropsSI("T", "P", p2, "H", h2, fluid)
    except Exception:
        T_out = T_line

    if T_sat is None:
        phase = "gas" if T_out >= T_line else "liq"
    else:
        phase = "gas" if T_out > (T_sat + 0.5) else "liq"

    return T_out, h2, phase


def nhne_out_state_from_mdot(
    fluid: str,
    p1: float,
    p2: float,
    T_line: float,
    D: float,
    mdot_nhne: float,
    h1_hint: Optional[float] = None,
    max_iter: int = 10000,
) -> Dict[str, Any]:
    """
    Given a target mass flow mdot (typically the phase-aware one),
    compute a self-consistent outlet state:

      - T_out, U_out, Mach, Re
      - rho_mix, mu_mix
      - x_out (mass quality), alpha_out (void fraction)
      - rho_l, rho_v
      - j_liq, j_gas (superficial phase velocities)
    """
    A = 0.25 * math.pi * D * D
    mdot = max(mdot_nhne, 0.0)

    # Upstream enthalpy h1
    if h1_hint is not None:
        h1 = float(h1_hint)
    else:
        try:
            h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
        except Exception:
            h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    # Initial "liquid" density at outlet
    try:
        rho2 = cp.PropsSI("D", "P", p2, "Q", 0, fluid)
    except Exception:
        T_sat_p2 = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
        rho2 = rho_singlephase_at_T(
            fluid,
            p2,
            max(T_sat_p2 - 0.5, 100.0),
            side="liq",
        )

    T_out = T_line
    rho_l = rho_v = None
    x = 0.0
    is_two = False

    for _ in range(max_iter):
        U_out = mdot / max(rho2 * A, 1e-12)
        T_out, h2, phase_hint = estimate_T_out_energy(
            fluid,
            p1,
            T_line,
            p2,
            U_out=U_out,
            U_in=0.0,
        )

        rho_mix, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p2, h2)

        if rho_mix is None:
            # Single-phase fallback
            try:
                T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
            except Exception:
                T_sat = T_out

            if T_out > T_sat + 0.5:
                rho_mix = rho_singlephase_at_T(
                    fluid,
                    p2,
                    T_out,
                    side="gas",
                )
                x, is_two = 1.0, False
            else:
                rho_mix = rho_singlephase_at_T(
                    fluid,
                    p2,
                    T_out,
                    side="liq",
                )
                x, is_two = 0.0, False

        # Density convergence
        if abs(rho_mix - rho2) <= 1e-3 * max(rho2, 1.0):
            rho2 = rho_mix
            break

        rho2 = rho_mix

    # Mixture viscosity
    if is_two:
        mu_l = _safe_viscosity(fluid, p2, phase="liq")
        mu_g = _safe_viscosity(fluid, p2, T_out, phase="gas")
        mu_mix = (1.0 - x) * mu_l + x * mu_g
    else:
        try:
            T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
        except Exception:
            T_sat = T_out
        phase_single = "gas" if T_out > T_sat + 0.5 else "liq"
        mu_mix = _safe_viscosity(fluid, p2, T_out, phase=phase_single)

    # Void fraction alpha
    if is_two and (rho_l is not None) and (rho_v is not None):
        Vv = x / max(rho_v, 1e-12)
        Vl = (1.0 - x) / max(rho_l, 1e-12)
        alpha_out = Vv / max(Vv + Vl, 1e-12)
    else:
        alpha_out = float(x >= 0.999)

    # Velocities and dimensionless numbers
    U_out = mdot / max(rho2 * A, 1e-12)
    try:
        T_sat_p2 = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat_p2 = T_out

    a_out = _safe_speed_of_sound(
        fluid,
        p2,
        T_out,
        two_phase=is_two,
        x=x,
        rho_l=rho_l,
        rho_v=rho_v,
    )
    Mach = U_out / max(a_out, 1e-9)
    Re = rho2 * U_out * D / max(mu_mix, 1e-12)

    # Phase volumetric flow rates
    Vdot = mdot / max(rho2, 1e-12)
    if is_two and (rho_l is not None) and (rho_v is not None):
        Vdot_v = x * mdot / max(rho_v, 1e-12)
        Vdot_l = (1.0 - x) * mdot / max(rho_l, 1e-12)
    else:
        if x >= 0.999:     # effectively all gas
            Vdot_v = Vdot
            Vdot_l = 0.0
        elif x <= 1e-12:   # effectively all liquid
            Vdot_l = Vdot
            Vdot_v = 0.0
        else:              # generic fallback
            Vdot_l = (1.0 - x) * Vdot
            Vdot_v = x * Vdot

    j_liq = Vdot_l / A
    j_gas = Vdot_v / A

    # Phase densities for output
    if is_two:
        rho_l_nhne = rho_l
        rho_v_nhne = rho_v
    else:
        if T_out > T_sat_p2 + 0.5:
            rho_v_nhne = rho_singlephase_at_T(fluid, p2, T_out, side="gas")
            rho_l_nhne = None
        else:
            rho_l_nhne = rho_singlephase_at_T(fluid, p2, T_out, side="liq")
            rho_v_nhne = None

    return dict(
        T_out=T_out,
        rho_mix=rho2,
        mu_mix=mu_mix,
        a_out=a_out,
        U_out=U_out,
        Mach=Mach,
        Re_out=Re,
        x_out=float(x),
        alpha_out=float(min(max(alpha_out, 0.0), 1.0)),
        Vdot_l=Vdot_l,
        Vdot_v=Vdot_v,
        j_liq=j_liq,
        j_gas=j_gas,
        rho_l=rho_l_nhne,
        rho_v=rho_v_nhne,
        phase_out=(
            "two-phase"
            if is_two
            else ("gas" if T_out > T_sat_p2 + 0.5 else "liquid")
        ),
    )


# ======================================================================
#                  DISCHARGE COEFFICIENT GEOMETRY MODEL
# ======================================================================
def _Re_from_mdot_target(
    mdot_target: float,
    D_orif: float,
    mu_out: float,
) -> float:
    """
    Reynolds number at the orifice using the target mass flow:

        Re = 4 * mdot / (π * μ * D)

    where mdot is the target mass flow and μ is the viscosity at the outlet
    state (liquid or near-liquid, as used in the core).
    """
    D_orif = max(D_orif, 1e-9)
    mu_out = max(mu_out, 1e-9)
    return float(4.0 * mdot_target / (math.pi * mu_out * D_orif))


def _gain_r_over_D(
    r_over_D: float,
    *,
    gain_max: float = 0.25,
    r_sat: float = 0.08,
) -> float:
    """
    Gain on Cd due to inlet rounding r/D, with asymptotic saturation:

        G_r = 1                              for r/D = 0  (sharp edge)
        G_r → 1 + gain_max ≈ 1.25            for r/D >> r_sat

    With gain_max = 0.25 and r_sat = 0.08:
      r/D ~ 0.20–0.25 → G_r ~ 1.23–1.24 (≈ +23–24%).
    """
    r_over_D = max(r_over_D, 0.0)
    gain_max = max(0.0, gain_max)

    G = 1.0 + gain_max * (1.0 - math.exp(-r_over_D / max(r_sat, 1e-6)))
    return float(G)


def _friction_factor_moody(
    Re: float,
    D_hyd: float,
    eps_abs: float,
) -> float:
    """
    Darcy friction factor f(Re, ε/D) in Moody-style:

      - Laminar:   f = 64 / Re
      - Turbulent: Swamee–Jain correlation (good fit to Colebrook–White).
    """
    Re = max(Re, 1.0)
    D_hyd = max(D_hyd, 1e-9)
    rel_rough = max(eps_abs / D_hyd, 1e-7)

    # Laminar flow
    if Re < 2300.0:
        return 64.0 / Re

    # Swamee–Jain for turbulent flow in rough pipes
    A = rel_rough / 3.7 + 5.74 / (Re ** 0.9)
    f = 0.25 / (math.log10(A) ** 2)
    return float(f)


def _K_darcy(
    Re_orifice: float,
    D_orif: float,
    L: float,
    eps_abs: float,
) -> float:
    """
    Darcy loss coefficient for the cylindrical part of the orifice:

        K_D = f(Re, ε/D) * (L / D)
    """
    D_orif = max(D_orif, 1e-9)
    L_over_D = max(L / D_orif, 0.0)

    f_D = _friction_factor_moody(Re_orifice, D_orif, eps_abs)
    return float(f_D * L_over_D)


def _Cd_reader_harris_gallagher(
    beta: float,
    Re_D: float,
    D_pipe: float,
    tap_type: str = "corner",
) -> float:
    """
    Discharge coefficient Cd for a sharp-edged orifice plate
    (Reader–Harris/Gallagher, ISO 5167).

    Parameters
    ----------
    beta : float
        Diameter ratio d/D (orifice / pipe).
    Re_D : float
        Reynolds number based on D (pipe).
    D_pipe : float
        Pipe diameter upstream [m].
    tap_type : {'corner', 'flange', 'D_D2'}
        Type of pressure tappings.
    """
    beta = float(np.clip(beta, 0.1, 0.75))
    Re_D = float(np.clip(Re_D, 4e3, 1e7))
    D_pipe = float(max(D_pipe, 1e-6))

    # Geometry of pressure tappings
    if tap_type == "corner":
        L1 = 0.0
        L2p = 0.0
    elif tap_type == "flange":
        L1 = 0.0254 / D_pipe
        L2p = L1
    elif tap_type == "D_D2":
        L1 = 1.0
        L2p = 0.47
    else:
        raise ValueError(f"Unknown tap_type: {tap_type}")

    A = (19000.0 * beta / Re_D) ** 0.8
    M2p = 2.0 * L2p / (1.0 - beta)

    term1 = 0.5961
    term2 = 0.0261 * beta**2
    term3 = -0.216 * beta**8
    term4 = 0.000521 * (1.0e6 * beta / Re_D) ** 0.7
    term5 = (0.0188 + 0.0063 * A) * beta**3.5 * (1.0e6 / Re_D) ** 0.3
    term6 = (
        0.043
        + 0.080 * math.exp(-10.0 * L1)
        - 0.123 * math.exp(-7.0 * L1)
    )
    term6 *= (1.0 - 0.11 * A) * beta**4 / (1.0 - beta**4)
    term7 = -0.031 * (M2p - 0.8 * M2p**1.1) * beta**1.3

    C = term1 + term2 + term3 + term4 + term5 + term6 + term7

    # Correction for small pipe diameters (D_pipe < 71.2 mm)
    if D_pipe < 0.0712:
        C += 0.011 * (0.75 - beta) * (2.8 - D_pipe / 0.0254)

    return float(np.clip(C, 0.3, 0.99))


def estimate_Cd_geom_full(
    D_orif: float,
    L: float,
    r_over_D: float,
    beta: float,
    mdot_target: float,
    rho_out: float,
    mu_out: float,
    D_pipe: float,
    eps_abs: float,
    *,
    tap_type: str = "corner",
) -> tuple[float, float]:
    """
    Full semi-empirical estimate of the discharge coefficient Cd:

      1) Cd_RHG(β, Re_D = β Re_or) for a sharp-edged plate orifice.
      2) Apply gain G_r(r/D) on Cd to account for rounded inlet.
      3) Convert to equivalent loss coefficient
         K_shape = 1/Cd^2 - 1 (form losses).
      4) Add Darcy losses along the orifice:
         K_Darcy = f(Re_or, ε/D) * (L/D).
      5) Convert back to Cd_final = 1 / sqrt(1 + K_tot).

    Main parameters:
      - D_orif, L, r_over_D : orifice geometry
      - beta                : diameter ratio D_orif / D_pipe
      - mdot_target         : required mass flow
      - rho_out, mu_out     : properties at outlet state
      - D_pipe              : upstream pipe diameter
      - eps_abs             : absolute roughness of the orifice
    """
    D_orif = max(D_orif, 1e-9)
    L = max(L, 0.0)
    mu_out = max(mu_out, 1e-9)

    # 1) Reynolds at orifice using the target mass flow
    Re_or = _Re_from_mdot_target(mdot_target, D_orif, mu_out)

    # Reynolds based on D_pipe for RHG (Re_D = β * Re_or)
    Re_D = beta * Re_or

    # 2) Base Cd (sharp-edged thin plate)
    Cd_plate = _Cd_reader_harris_gallagher(beta, Re_D, D_pipe, tap_type=tap_type)

    # 3) Gain due to inlet rounding r/D applied to plate Cd
    G_r = _gain_r_over_D(r_over_D)
    Cd_eq = Cd_plate * G_r   # "equivalent" Cd without friction along L

    # 4) Equivalent form losses with rounding
    K_shape = 1.0 / (Cd_eq**2) - 1.0

    # 5) Darcy losses along the cylindrical orifice
    K_d = _K_darcy(Re_or, D_orif, L, eps_abs)

    # 6) Total K and final Cd
    K_tot = K_shape + K_d
    Cd_final = 1.0 / math.sqrt(1.0 + K_tot)

    # Safety clamp
    Cd_final = float(np.clip(Cd_final, 0.3, 0.95))

    return Cd_final, Re_or


def estimate_Cd_from_geometry(
    fluid: str,
    p1: float,
    T_line: float,
    D_orif: float,
    L: float,
    r_over_D: float,
    D_pipe: float,
    mdot_target: float,
    *,
    Cd_input: float | None = None,
    eps_abs: float = 2e-6,
) -> tuple[float, float, float, float]:
    """
    Geometry-based Cd estimate wrapper.

    If Cd_input is not None and > 0, it is used directly as Cd and only
    Re is computed for diagnostics. Otherwise the RHG + r/D + Darcy model
    is used to estimate Cd.

    Returns:
      (Cd_used, Re_char, rho_out, mu_out)
    """
    # Outlet-state properties: can be replaced by actual core outlet values
    rho_out = rho_singlephase_at_T(fluid, p1, T_line, side="liq")
    mu_out = _safe_viscosity(fluid, p1, T_line, phase="liq")

    beta = D_orif / D_pipe

    if Cd_input is not None and Cd_input > 0.0:
        Cd_used = float(Cd_input)
        # Re still computed for logging/diagnostics
        Re_char = _Re_from_mdot_target(mdot_target, D_orif, mu_out)
    else:
        Cd_used, Re_char = estimate_Cd_geom_full(
            D_orif=D_orif,
            L=L,
            r_over_D=r_over_D,
            beta=beta,
            mdot_target=mdot_target,
            rho_out=rho_out,
            mu_out=mu_out,
            D_pipe=D_pipe,
            eps_abs=eps_abs,
            tap_type="corner",
        )

    return Cd_used, Re_char, rho_out, mu_out