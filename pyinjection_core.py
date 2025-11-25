"""
pyinjection_core.py
-------------------

Core backend per l'iniettore plain-orifice N2O:

- Modelli di portata:
    * SPI (single-phase incompressible/compressible)
    * HEM
    * NHNE = blend SPI/HEM
- Selettore phase-aware (SPI vs NHNE)
- Stato in uscita coerente con mdot (HEM + bilancio entalpia di ristagno)
- Helper per proprietà termodinamiche robusti (vicino a saturazione)
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any

import CoolProp.CoolProp as cp

# ----------------------------------------------------------------------
# Parametri di guardia
# ----------------------------------------------------------------------
DELTA_T_HYST = 0.5          # [K] isteresi per confronti con Tsat
EPS_REL_PSAT = 1e-4         # tolleranza relativa su |P - Psat|/Psat
P_EPS        = 5.0e4        # [Pa] eps su denominatori pressione
PKEY_DPA     = 1.0e2        # [Pa] quantizzazione pressione

# fallback fisici per proprietà
MU_GAS_FALLBACK: float   = 1.85e-5   # Pa·s
MU_LIQ_FALLBACK: float   = 3.00e-4   # Pa·s
AOUT_GAS_FALLBACK: float = 203.44    # m/s (fallback a_out, lato gas)

# ----------------------------------------------------------------------
# Utilità di saturazione robuste
# ----------------------------------------------------------------------
def _pkey(p: float) -> float:
    """Quantizza la pressione per stabilizzare vicino a saturazione."""
    return float(round(p / PKEY_DPA) * PKEY_DPA)


def _safe_psat_at_T(fluid: str, T: float) -> float:
    """
    Psat(T) robusto. Se non definito (sopra Tc o errore), fallback a pcrit
    o 1 bar.
    """
    try:
        return cp.PropsSI("P", "T", T, "Q", 1, fluid)
    except Exception:
        try:
            return cp.PropsSI("pcrit", fluid)
        except Exception:
            return 1.0e5  # fallback estremo 1 bar


def _safe_tcrit_pcrit(fluid: str) -> Tuple[Optional[float], Optional[float]]:
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
    Tsat(P) robusto: T tale che Q=1 a P; se sopra-critico o errore → None.
    """
    Tc, Pc = _safe_tcrit_pcrit(fluid)
    if (Pc is not None) and (p >= Pc):
        return None
    try:
        return cp.PropsSI("T", "P", _pkey(p), "Q", 1, fluid)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Helper densità monofase vicino a saturazione
# ----------------------------------------------------------------------
def _rho_single_at_T(fluid: str, p: float, T: float, side: str) -> float:
    """
    Densità monofase robusta vicino a saturazione.
    Se |p - Psat(T)|/Psat(T) < EPS_REL_PSAT -> sposta p lato 'side'.
    """
    try:
        p_sat = _safe_psat_at_T(fluid, T)
        if abs(p - p_sat) / max(p_sat, 1.0) < EPS_REL_PSAT:
            p = (0.999 * p_sat) if (side == "gas") else (1.001 * p_sat)
    except Exception:
        pass
    return cp.PropsSI("D", "P", _pkey(p), "T", T, fluid)


# ----------------------------------------------------------------------
# Blend factor k per NHNE
# ----------------------------------------------------------------------
def _k_local(fluid: str,
             p_local: float,
             p2: float,
             T_line: float,
             L_over_D: Optional[float] = None,
             K_RESIDENCE: float = 0.0) -> float:
    """
    k = sqrt( (p_local - p2) / max(Psat(T_line) - p2, P_EPS) ) smorzato da L/D.

    K_RESIDENCE permette di ridurre k per fori più lunghi (maggiore permanenza).
    """
    pV = _safe_psat_at_T(fluid, T_line)
    den = max(pV - p2, P_EPS)
    num = max(p_local - p2, 0.0)
    k = math.sqrt(num / den)
    if L_over_D and (L_over_D > 0.0) and (K_RESIDENCE > 0.0):
        k /= (1.0 + K_RESIDENCE * L_over_D)
    return k


# ======================================================================
#                         MODELLI DI PORTATA
# ======================================================================
def _mdot_spi(fluid: str,
              p1_bar: float, p2_bar: float,
              T_line: float,
              D: float, Cd: float,
              use_compress: bool = True,
              n_isentropic: Optional[float] = None) -> float:
    """
    Modello SPI con:
      - scelta fase a monte (gas vs liquido) da T_line vs Tsat(P1)
      - opzionale correzione di comprimibilità Y'

    Ritorna mdot_SPI [kg/s].
    """
    if D <= 0.0 or Cd <= 0.0 or p1_bar <= p2_bar:
        return 0.0

    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)
    A = 0.25 * math.pi * D * D
    dp = p1 - p2

    # Fase a monte
    Tc, _ = _safe_tcrit_pcrit(fluid)
    try:
        T_sat1 = cp.PropsSI("T", "P", p1, "Q", 1, fluid)
        upstream_is_gas = (T_line > T_sat1 + DELTA_T_HYST)
    except Exception:
        upstream_is_gas = (Tc is not None) and (T_line >= Tc)

    rho_ref = _rho_single_at_T(fluid, p1, T_line,
                               side=("gas" if upstream_is_gas else "liq"))

    mdot_spi = Cd * A * math.sqrt(max(2.0 * rho_ref * dp, 0.0))
    if not use_compress:
        return mdot_spi

    # correzione comprimibilità
    if n_isentropic and (n_isentropic > 1.0):
        pr = max(p2 / p1, 1e-12)
        Yp = (n_isentropic / (n_isentropic - 1.0)) * (
            1.0 - pr ** ((n_isentropic - 1.0) / n_isentropic)
        )
    else:
        a1 = cp.PropsSI("A", "P", p1, "T", T_line, fluid)
        K  = max(rho_ref * a1 * a1, 1e5)
        Yp = dp / K

    return mdot_spi / math.sqrt(max(1.0 + Yp, 1e-6))


def _mdot_hem(fluid: str,
              p1_bar: float, p2_bar: float,
              T_line: float,
              D: float, Cd: float) -> float:
    """
    HEM energetico minimale (solo mdot) con guardie:

    - h1 da (P1, T_line) o da Q=0
    - iterazione su h2 e rho(P2, H2)
    """
    if D <= 0.0 or Cd <= 0.0 or p1_bar <= p2_bar:
        return 0.0

    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)
    A = 0.25 * math.pi * D * D

    try:
        h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
    except Exception:
        h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    # start liquido a valle
    try:
        rho0 = cp.PropsSI("D", "P", p2, "Q", 0, fluid)
    except Exception:
        rho0 = _rho_single_at_T(fluid, p2, T_line, side="liq")

    mdot = Cd * A * math.sqrt(max(2.0 * rho0 * (p1 - p2), 0.0))

    for _ in range(1000):
        U   = mdot / max(rho0 * A, 1e-12)
        h2  = h1 - 0.5 * U * U

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

    return max(mdot, 0.0)


def _mdot_nhne(fluid: str,
               p1_bar: float, p2_bar: float,
               T_line: float,
               D: float, Cd: float,
               L_over_D: Optional[float] = None,
               K_RESIDENCE: float = 0.0,
               use_spi_compress: bool = True,
               spi_n: Optional[float] = None) -> Tuple[float, float]:
    """
    NHNE = blend(k/(k+1))*SPI + (1/(k+1))*HEM
    Ritorna (mdot_nhne, k).
    """
    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)

    mdot_spi = _mdot_spi(
        fluid, p1_bar, p2_bar, T_line, D, Cd,
        use_compress=use_spi_compress, n_isentropic=spi_n
    )
    mdot_hem = _mdot_hem(fluid, p1_bar, p2_bar, T_line, D, Cd)

    k = _k_local(
        fluid, p_local=p1, p2=p2, T_line=T_line,
        L_over_D=L_over_D, K_RESIDENCE=K_RESIDENCE
    )

    mdot_nhne = (k / (k + 1.0)) * mdot_spi + (1.0 / (k + 1.0)) * mdot_hem
    return mdot_nhne, k


# ----------------------------------------------------------------------
# Classificatore minimale per decidere se SPI produce bifase
# ----------------------------------------------------------------------
def _phase_from_spi_guess(fluid: str,
                          p1: float, p2: float,
                          T_line: float,
                          D: float,
                          mdot_spi: float) -> str:
    """
    Decide se con mdot_SPI l'uscita sarebbe bifase usando bilancio h0
    e banda di saturazione.

    Ritorna: 'two-phase' | 'liquid' | 'gas'
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

    U  = mdot_spi / max(rho2 * A, 1e-12)
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

    # monofase: decidi gas/liquido
    try:
        T_out = cp.PropsSI("T", "P", p2, "H", h2, fluid)
    except Exception:
        T_out = T_line

    if have_sat:
        return "gas" if (T_out > T_sat2 + DELTA_T_HYST) else "liquid"
    else:
        return "gas" if (T_out >= T_line) else "liquid"


# ----------------------------------------------------------------------
# API phase-aware
# ----------------------------------------------------------------------
def compute_mdot_phaseaware(fluid: str,
                            p1_bar: float, p2_bar: float,
                            T_line: float,
                            D: float, Cd: float,
                            *,
                            L: Optional[float] = None,
                            use_spi_compress: bool = True,
                            spi_n: Optional[float] = None,
                            K_RESIDENCE: float = 0.0) -> Tuple[float, str]:
    """
    Ritorna (mdot_phaseaware, model_used), con scelta automatica:
      - se con mdot_SPI l'uscita è bifase -> usa NHNE
      - altrimenti usa SPI
    """
    p1, p2 = _pkey(p1_bar * 1e5), _pkey(p2_bar * 1e5)
    L_over_D = (L / D) if (L is not None and D > 0.0) else None

    mdot_spi = _mdot_spi(
        fluid, p1_bar, p2_bar, T_line, D, Cd,
        use_compress=use_spi_compress, n_isentropic=spi_n
    )
    mdot_nhne, _ = _mdot_nhne(
        fluid, p1_bar, p2_bar, T_line, D, Cd,
        L_over_D=L_over_D, K_RESIDENCE=K_RESIDENCE,
        use_spi_compress=use_spi_compress, spi_n=spi_n
    )

    phase_spi = _phase_from_spi_guess(fluid, p1, p2, T_line, D, mdot_spi)

    if phase_spi == "two-phase":
        return mdot_nhne, "NHNE"
    else:
        return mdot_spi, "SPI"


# ======================================================================
#                       BLOCCO PROPRIETÀ DI USCITA
# ======================================================================
def _safe_viscosity(fluid: str, p: float, T: Optional[float] = None,
                    phase: str = "gas", x: Optional[float] = None) -> float:
    """
    Viscosità robusta:
      - phase = 'gas' / 'liq' -> singola fase
      - altri valori -> blend due-fasi (HEM) in base a x
    """
    try:
        if phase == "gas":
            if T is None:
                return MU_GAS_FALLBACK
            return cp.PropsSI("V", "P", p, "T", T, fluid)

        if phase == "liq":
            if T is not None:
                try:
                    p_sat = cp.PropsSI("P", "T", T, "Q", 1, fluid)
                    p_eff = max(p, 1.001 * p_sat)
                except Exception:
                    p_eff = p
                return cp.PropsSI("V", "P", p_eff, "T", T, fluid)
            return cp.PropsSI("V", "P", p, "Q", 0, fluid)

        # due-fasi: blend HEM
        xx   = 0.5 if (x is None) else float(min(max(x, 0.0), 1.0))
        mu_l = _safe_viscosity(fluid, p, T=None, phase="liq")
        mu_g = _safe_viscosity(fluid, p, T=T, phase="gas") if T is not None \
               else _safe_viscosity(fluid, p, phase="gas")
        return (1.0 - xx) * mu_l + xx * mu_g

    except Exception:
        return MU_GAS_FALLBACK if phase == "gas" else MU_LIQ_FALLBACK


def _safe_speed_of_sound(fluid: str,
                         p: float,
                         T: Optional[float] = None,
                         two_phase: bool = False,
                         x: Optional[float] = None,
                         rho_l: Optional[float] = None,
                         rho_v: Optional[float] = None) -> float:
    """
    Velocità del suono robusta:
      - per ora: se two_phase -> usa lato gas a T_sat(P) (o T_out)
      - altrimenti: a(P,T) monofase
    """
    try:
        if two_phase:
            # uso lato gas come approssimazione
            T_sat = cp.PropsSI("T", "P", p, "Q", 1, fluid)
            T_use = max(T or (T_sat + 0.5), T_sat + 0.5)
            return cp.PropsSI("A", "P", p, "T", T_use, fluid)

        if T is None:
            return AOUT_GAS_FALLBACK

        return cp.PropsSI("A", "P", p, "T", T, fluid)

    except Exception:
        return AOUT_GAS_FALLBACK


def rho_singlephase_at_T(fluid: str,
                         p: float,
                         T: float,
                         side: str = "gas") -> float:
    """
    Densità robusta vicino a saturazione: forza 'gas' o 'liq'
    spostando leggermente P dal Psat(T).
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


def mixture_rho_HEM(fluid: str, p: float, h: float) -> Tuple[
    Optional[float], Optional[float], Optional[float], float, bool
]:
    """
    HEM a (p,h): qualità x e densità miscela.
    Ritorna (rho_mix, rho_l, rho_v, x, is_two).
    """
    try:
        h_f  = cp.PropsSI("H", "P", p, "Q", 0, fluid)
        h_g  = cp.PropsSI("H", "P", p, "Q", 1, fluid)
        rho_l= cp.PropsSI("D", "P", p, "Q", 0, fluid)
        rho_v= cp.PropsSI("D", "P", p, "Q", 1, fluid)

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


def estimate_T_out_energy(fluid: str,
                          p1: float,
                          T_line: float,
                          p2: float,
                          U_out: float,
                          U_in: float = 0.0) -> Tuple[float, float, str]:
    """
    Stima T_out da bilancio di entalpia di ristagno:
      h2 = h1 + U_in^2/2 - U_out^2/2

    Ritorna (T_out, h2, phase_hint) dove phase_hint appartiene a {'gas', 'liq', 'two_phase'}.
    """
    # h1 a monte
    try:
        h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
    except Exception:
        h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    h2 = h1 + 0.5 * (U_in * U_in - U_out * U_out)
    h2 = max(h2, h1 - 1e7)

    # Saturazione a valle
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
        h_f2  = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        h_g2  = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = None
        h_f2 = h_g2 = None

    # two-phase?
    if (h_f2 is not None) and (h_g2 is not None) and (h_f2 <= h2 <= h_g2):
        return (T_sat if T_sat is not None else T_line), h2, "two_phase"

    # monofase
    try:
        T_out = cp.PropsSI("T", "P", p2, "H", h2, fluid)
    except Exception:
        T_out = T_line

    if T_sat is None:
        phase = "gas" if T_out >= T_line else "liq"
    else:
        phase = "gas" if T_out > (T_sat + 0.5) else "liq"

    return T_out, h2, phase


def nhne_out_state_from_mdot(fluid: str,
                             p1: float, p2: float,
                             T_line: float,
                             D: float,
                             mdot_nhne: float,
                             h1_hint: Optional[float] = None,
                             max_iter: int = 10000) -> Dict[str, Any]:
    """
    Dato mdot (tipicamente quello phase-aware), rende consistenti le
    grandezze di uscita:

      - T_out, U_out, Mach, Re
      - rho_mix, mu_mix
      - x_out (qualità massica), alpha_out (frazione volumetrica)
      - rho_l, rho_v
      - j_liq, j_gas (velocità superficiali)
    """
    A   = 0.25 * math.pi * D * D
    mdot = max(mdot_nhne, 0.0)

    # h1 a monte
    if h1_hint is not None:
        h1 = float(h1_hint)
    else:
        try:
            h1 = cp.PropsSI("H", "P", p1, "T", T_line, fluid)
        except Exception:
            h1 = cp.PropsSI("H", "P", p1, "Q", 0, fluid)

    # init: densità "liquida" a p2
    try:
        rho2 = cp.PropsSI("D", "P", p2, "Q", 0, fluid)
    except Exception:
        T_sat_p2 = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
        rho2 = rho_singlephase_at_T(fluid, p2, max(T_sat_p2 - 0.5, 100.0),
                                    side="liq")

    T_out = T_line
    rho_l = rho_v = None
    x = 0.0
    is_two = False

    for _ in range(max_iter):
        U_out = mdot / max(rho2 * A, 1e-12)
        T_out, h2, phase_hint = estimate_T_out_energy(
            fluid, p1, T_line, p2, U_out=U_out, U_in=0.0
        )

        rho_mix, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p2, h2)

        if rho_mix is None:
            # monofase
            try:
                T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
            except Exception:
                T_sat = T_out

            if T_out > T_sat + 0.5:
                rho_mix = rho_singlephase_at_T(fluid, p2, T_out, side="gas")
                x, is_two = 1.0, False
            else:
                rho_mix = rho_singlephase_at_T(fluid, p2, T_out, side="liq")
                x, is_two = 0.0, False

        # convergenza densità
        if abs(rho_mix - rho2) <= 1e-3 * max(rho2, 1.0):
            rho2 = rho_mix
            break

        rho2 = rho_mix

    # viscosità miscela
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

    # frazione volumetrica alpha
    if is_two and (rho_l is not None) and (rho_v is not None):
        Vv = x / max(rho_v, 1e-12)
        Vl = (1.0 - x) / max(rho_l, 1e-12)
        alpha_out = Vv / max(Vv + Vl, 1e-12)
    else:
        alpha_out = float(x >= 0.999)

    # velocità e numeri adimensionali
    U_out = mdot / max(rho2 * A, 1e-12)
    try:
        T_sat_p2 = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat_p2 = T_out

    a_out = _safe_speed_of_sound(
        fluid, p2, T_out,
        two_phase=is_two, x=x, rho_l=rho_l, rho_v=rho_v
    )
    Mach  = U_out / max(a_out, 1e-9)
    Re    = rho2 * U_out * D / max(mu_mix, 1e-12)

    # portate volumetriche di fase
    Vdot = mdot / max(rho2, 1e-12)
    if is_two and (rho_l is not None) and (rho_v is not None):
        Vdot_v = x * mdot / max(rho_v, 1e-12)
        Vdot_l = (1.0 - x) * mdot / max(rho_l, 1e-12)
    else:
        if x >= 0.999:     # gas
            Vdot_v = Vdot
            Vdot_l = 0.0
        elif x <= 1e-12:   # liquido
            Vdot_l = Vdot
            Vdot_v = 0.0
        else:              # fallback
            Vdot_l = (1.0 - x) * Vdot
            Vdot_v = x * Vdot

    j_liq = Vdot_l / A
    j_gas = Vdot_v / A

    # densità di fase per stampa
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
        phase_out=("two-phase"
                   if is_two
                   else ("gas" if T_out > T_sat_p2 + 0.5 else "liquid")),
    )

# =============================================================================
# Discharge coefficient model for short-orifice injectors
# =============================================================================
#
# Fonti sperimentali principali usate per calibrare Cd(r/D, L/D, Re):
#
# [1] Reader–Harris, M.J., Gallagher, P.M. (1998).
#     "The Orifice Plate Discharge Coefficient Equation".
#     Base ISO 5167 per Cd(Re, β) in orifizi a spigolo vivo.
#
# [2] Gelalles, A.G., Marsh, E.T. (1931).
#     "Effect of Orifice Length-Diameter Ratio on the Coefficient of Discharge".
#     Dati storici su L/D in orifizi corti cilindrici.
#
# [3] Edlebeck, S. (2013).
#     "Measurements of the Flow of Supercritical CO₂ Through Short Orifices".
#     Effetti di L/D elevati e regime trans-/supercritico.
#
# [4] Waxman, J., Dyer, J., Karabeyoglu, A. (2019, Stanford).
#     Misure di Cd per iniettori a orifizio singolo per N₂O,
#     con varianti a spigolo vivo / smussato / raccordato.
#
# [5] Dataset interno fornito dall'utente:
#     "Coefficiente di Scarico (Cd) in Orifizi Corti per N₂O (e Fluidi Simili) – Dataset e Correlazioni.pdf".
#     → Usato per allineare i range numerici alle condizioni tipiche N₂O
#       nel campo 10⁴ ≲ Re ≲ 10⁵ e 0.5 ≲ L/D ≲ 10.
#
# Il modello sottostante non è una correlazione diretta di un singolo paper,
# ma una sintesi semplificata e regolarizzata delle tendenze sperimentali.
# Serve come stima di primo livello coerente con la letteratura.
# =============================================================================

def estimate_Cd_geom(r_over_D: float,
                     L_over_D: float,
                     Re: float) -> float:
    """Correlazione geometrica sintetica Cd(r/D, L/D, Re)."""
    r_over_D = max(r_over_D, 0.0)
    L_over_D = max(L_over_D, 0.1)
    Re       = max(Re, 1.0)

    # Cd base per spigolo vivo (funzione di Re)
    Re_grid   = np.array([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5])
    Cd_sharp  = np.array([0.58, 0.60, 0.62, 0.63, 0.63, 0.63, 0.63])
    Cd_base   = float(np.interp(Re, Re_grid, Cd_sharp))

    # Fattore correttivo r/D
    rD_grid = np.array([0.00, 0.05, 0.10, 0.20, 0.30])
    f_rD    = np.array([1.00, 0.95, 1.10, 1.20, 1.25])
    f_r     = float(np.interp(r_over_D, rD_grid, f_rD))

    # Fattore correttivo L/D
    LD_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    f_LD    = np.array([0.96, 0.98, 1.00, 1.00, 0.97, 0.94, 0.90])
    f_ld    = float(np.interp(L_over_D, LD_grid, f_LD))

    Cd_est = Cd_base * f_r * f_ld
    return max(0.55, min(Cd_est, 0.90))


def estimate_Cd_from_geometry(
    fluid: str,
    p1: float,
    T_line: float,
    D: float,
    L: float,
    r_over_D: float,
    *,
    Cd_input: float | None = None,
    Re_char: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Helper high-level per ricavare Cd a partire dalla geometria dell'orifizio,
    con opzionale override da dati CFD/esperimento.

    Parametri
    ---------
    fluid : str
        Nome CoolProp del fluido.
    p1 : float
        Pressione a monte [Pa].
    T_line : float
        Temperatura linea a monte [K].
    D, L : float
        Diametro e lunghezza orifizio [m].
    r_over_D : float
        Edge radius ratio (raccordo d'ingresso).
    Cd_input : float | None
        Se >0, questo valore viene usato direttamente (es. da CFD/esperimenti).
        Se None o <=0, si usa la correlazione geometrica estimate_Cd_geom.
    Re_char : float | None
        Reynolds caratteristico. Se None, viene stimato a partire da
        rho_l, mu_l e una velocità caratteristica fittizia.

    Ritorna
    -------
    Cd_used : float
        Cd effettivamente usato.
    Re_char : float
        Reynolds caratteristico utilizzato nella correlazione.
    rho_l : float
        Densità liquida di riferimento a monte [kg/m^3].
    mu_l : float
        Viscosità liquida di riferimento a monte [Pa*s].
    """
    # Proprietà monofase liquide di riferimento
    rho_l = rho_singlephase_at_T(fluid, p1, T_line, side="liq")
    mu_l  = _safe_viscosity(fluid, p1, T_line, phase="liq")

    if Re_char is None:
        # Velocità caratteristica fittizia (ordine di grandezza)
        U_char = 10.0
        Re_char = rho_l * U_char * D / max(mu_l, 1e-9)

    if Cd_input is not None and Cd_input > 0.0:
        Cd_used = float(Cd_input)
    else:
        L_over_D = L / D
        Cd_used  = estimate_Cd_geom(r_over_D, L_over_D, Re_char)

    return Cd_used, Re_char, rho_l, mu_l
