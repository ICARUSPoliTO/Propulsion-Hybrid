"""
pyinjection_performance.py
--------------------

Single-case plain-orifice N2O injector:
- uses the physical/thermodynamic backend in pyinjection_core.py
- can use a user-imposed Cd or estimate it from geometry
  through the geometric model in the core
- prints input tables, mass flow rates and outlet properties.

This file does not contain the physics of the models; it is only
a wrapper for output formatting.
"""

import math
import argparse
from typing import Dict, Any, List, Tuple
import CoolProp.CoolProp as cp
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# ============================
#  Simple tooltip for widgets
# ============================
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show_tip)
        widget.bind("<Leave>", self._hide_tip)

    def _show_tip(self, event=None):
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 8),
        )
        label.pack(ipadx=4, ipady=2)

    def _hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw is not None:
            tw.destroy()


# ============================================================
#  IMPORTS FROM CORE
# ============================================================

from pyinjection_core import (
    _mdot_spi,
    _mdot_hem,
    _mdot_nhne,
    compute_mdot_phaseaware,
    nhne_out_state_from_mdot,
    _safe_viscosity,
    rho_singlephase_at_T,
    estimate_Cd_from_geometry,  # geometric Cd (already uses rho_singlephase_at_T, _safe_viscosity, etc.)
)

# ============================================================
#  WRAPPER FOR Cd ESTIMATION IN THE PERFORMANCE TOOL
# ============================================================

def estimate_Cd_from_geometry_perf(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    D: float,
    L: float,
    r_over_D: float,
) -> tuple[float, float]:
    """
    Local wrapper to use estimate_Cd_from_geometry from the core
    without asking the user for D_pipe and mdot_target.

    - p1_bar, p2_bar in [bar]
    - Assumes D_pipe = D (single orifice on equivalent pipe)
    - mdot_target estimated with ideal orifice (Cd≈1) to build Re.
    """
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5

    # Single-phase properties in the line (already used by the core)
    rho_out = rho_singlephase_at_T(fluid, p1, T_line, side="liq")

    # Pressure drop for target mass flow estimation
    dp = max(p1 - p2, 1.0)  # avoids dp=0

    # Orifice area
    A = 0.25 * math.pi * D * D

    # "Target" ideal mass flow (Cd=1, incompressible orifice)
    mdot_target = rho_out * A * math.sqrt(2.0 * dp / max(rho_out, 1e-12))

    # Upstream pipe diameter: assumed equal to orifice
    D_pipe = D

    # Call to the core function
    Cd_used, Re_char, _, _ = estimate_Cd_from_geometry(
        fluid=fluid,
        p1=p1,
        T_line=T_line,
        D_orif=D,
        L=L,
        r_over_D=r_over_D,
        D_pipe=D_pipe,
        mdot_target=mdot_target,
        # Cd_input=None => actually use geometric+Darcy model
        Cd_input=None,
    )

    return Cd_used, Re_char


# ============================================================
#  TABLE PRINTING UTILITIES
# ============================================================

def _print_table(title: str,
                 columns: List[Tuple[str, str, int, str]],
                 rows: List[Dict[str, Any]]) -> None:
    """
    columns: list of (header, key, width, fmt)
             fmt is only for numbers (e.g. '.3f', '.3e'); ignored for strings.
    rows   : list of dict {key: value}
    """
    print("\n" + title)
    header = " | ".join(f"{h:>{w}}" for (h, _, w, _) in columns)
    sep    = "-+-".join("-" * w for (_, _, w, _) in columns)
    print(header)
    print(sep)

    for r in rows:
        cells = []
        for (_, key, w, fmt) in columns:
            val = r.get(key, "")
            if isinstance(val, (int, float)):
                cells.append(f"{val:>{w}{fmt}}")
            else:
                cells.append(f"{str(val):>{w}}")
        print(" | ".join(cells))
    print("")


# ============================================================
#  INPUT TABLE
# ============================================================

def print_inputs_table(params: Dict[str, Any]) -> None:
    """
    Print the input table using:
    - fluid, T_line, P1, P2
    - geometry: D, L, L/D, r/D, r, area A
    - Cd, mode and source (user / geometric).
    """
    cols = [
        ("Parameter", "k", 24, "s"),
        ("Value",     "v", 24, "s"),
    ]

    D = params["D"]
    L = params["L"]
    A = 0.25 * math.pi * D**2
    L_over_D = L / D if D > 0.0 else float("nan")

    r_over_D = params.get("r_over_D", None)
    if r_over_D is not None:
        r_edge = r_over_D * D
    else:
        r_edge = None

    rows = [
        {"k": "Fluid",            "v": params["fluid"]},
        {"k": "T_line [K]",       "v": f'{params["T_line"]:.3f}'},
        {"k": "P1 [bar]",         "v": f'{params["p1_bar"]:.3f}'},
        {"k": "P2 [bar]",         "v": f'{params["p2_bar"]:.3f}'},
        {"k": "Orifice D [m]",    "v": f'{D:.6f}'},
        {"k": "Orifice L [m]",    "v": f'{L:.6f}'},
        {"k": "L/D [-]",          "v": f'{L_over_D:.3f}'},
    ]

    # Edge radius geometry if available
    if r_over_D is not None:
        rows.append({"k": "r/D [-]",         "v": f'{r_over_D:.4f}'})
        rows.append({"k": "Edge radius r [m]", "v": f'{r_edge:.6f}'})

    rows.extend([
        {"k": "Orifice area A [m^2]", "v": f'{A:.8e}'},
        {"k": "Cd [-]",               "v": f'{params["Cd"]:.4f}'},
    ])

    # Cd mode and source (user / geometric)
    if "Cd_mode" in params:
        rows.append({"k": "Cd mode", "v": params["Cd_mode"]})
    if "Cd_source" in params:
        rows.append({"k": "Cd source", "v": params["Cd_source"]})

    _print_table("INPUT TABLE - Case parameters", cols, rows)


# ============================================================
#  MASS FLOW TABLE
# ============================================================

def print_mdot_table(mdot_spi: float,
                     mdot_hem: float,
                     mdot_nhne: float,
                     mdot_pa: float,
                     model_used: str) -> None:
    cols = [
        ("Model",        "model", 16, "s"),
        ("mdot [kg/s]",  "mdot",  18, ".6f"),
    ]
    rows = [
        {"model": "SPI",          "mdot": mdot_spi},
        {"model": "HEM",          "mdot": mdot_hem},
        {"model": "NHNE",         "mdot": mdot_nhne},
        {"model": "Phase-aware",  "mdot": mdot_pa},
    ]
    _print_table("MASS FLOW TABLE – SPI / HEM / NHNE / Phase-aware", cols, rows)
    print(f"Model automatically selected (phase-aware): {model_used}\n")


# ============================================================
#  OUTLET PHASE PROPERTIES TABLE
# ============================================================

def print_phase_properties_table(fluid: str,
                                 p2_bar: float,
                                 D: float,
                                 out: Dict[str, Any]) -> None:
    """
    Outlet properties for each phase:
    - if two-phase: liquid + vapor (same T_out, T_sat, but j_liq/j_gas, rho, mu, Re)
    - if single phase: single row for the present phase.

    For density and viscosity we use core utilities
    (rho_singlephase_at_T, _safe_viscosity); CoolProp is used only
    for T_sat, Cp, k, h.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    phase = out["phase_out"]
    T_out = out["T_out"]

    cols = [
        ("Phase",        "phase",  10, "s"),
        ("T [K]",        "T",       8, ".3f"),
        ("T_sat(P2)",    "Tsat",   10, ".3f"),
        ("U [m/s]",      "U",      10, ".3f"),
        ("rho [kg/m3]",  "rho",    12, ".3f"),
        ("mu [Pa·s]",    "mu",     12, ".3e"),
        ("Re [-]",       "Re",     12, ".3e"),
        ("alpha [-]",    "alpha",   8, ".3f"),
        ("Cp [J/kg/K]",  "cp",     12, ".3e"),
        ("MW [kg/kmol]", "MW",     12, ".3f"),
        ("h [J/kg]",     "h",      14, ".3e"),
    ]
    rows: List[Dict[str, Any]] = []

    # Overall vapor volume fraction
    alpha_mix = float(out.get("alpha_out", 0.0))

    # MW is the same for both phases
    try:
        MW_val = cp.PropsSI("M", fluid) * 1000.0  # kg/kmol
    except Exception:
        MW_val = float("nan")

    if phase == "two-phase":
        rho_l = out["rho_l"]
        rho_v = out["rho_v"]
        j_l   = out["j_liq"]
        j_v   = out["j_gas"]

        # Viscosities from core (with T_out)
        mu_l  = _safe_viscosity(fluid, p2, T_out, phase="liq")
        mu_v  = _safe_viscosity(fluid, p2, T_out, phase="gas")

        Re_l  = rho_l * j_l * D / max(mu_l, 1e-12) if rho_l is not None else 0.0
        Re_v  = rho_v * j_v * D / max(mu_v, 1e-12) if rho_v is not None else 0.0

        # Thermal properties at saturated conditions (liquid/vapor)
        try:
            cp_l = cp.PropsSI("C", "P", p2, "Q", 0, fluid)
        except Exception:
            cp_l = float("nan")
        try:
            cp_v = cp.PropsSI("C", "P", p2, "Q", 1, fluid)
        except Exception:
            cp_v = float("nan")
        try:
            h_l = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        except Exception:
            h_l = float("nan")
        try:
            h_v = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
        except Exception:
            h_v = float("nan")

        # Liquid volume fraction = 1 - alpha_mix
        rows.append(dict(
            phase="liquid",
            T=T_out, Tsat=T_sat,
            U=j_l, rho=rho_l, mu=mu_l, Re=Re_l,
            alpha=(1.0 - alpha_mix),
            cp=cp_l, MW=MW_val, h=h_l,
        ))
        # Vapor volume fraction = alpha_mix
        rows.append(dict(
            phase="vapor",
            T=T_out, Tsat=T_sat,
            U=j_v, rho=rho_v, mu=mu_v, Re=Re_v,
            alpha=alpha_mix,
            cp=cp_v, MW=MW_val, h=h_v,
        ))

    else:
        # Single phase: compute properties at (P2, T_out)
        if phase == "gas":
            rho = out["rho_v"] if out["rho_v"] is not None else rho_singlephase_at_T(
                fluid, p2, T_out, side="gas"
            )
            mu   = _safe_viscosity(fluid, p2, T_out, phase="gas")
            name = "gas"
            alpha_single = 1.0
        else:
            rho = out["rho_l"] if out["rho_l"] is not None else rho_singlephase_at_T(
                fluid, p2, T_out, side="liq"
            )
            mu   = _safe_viscosity(fluid, p2, T_out, phase="liq")
            name = "liquid"
            alpha_single = 0.0

        U  = out["U_out"]
        Re = rho * U * D / max(mu, 1e-12)

        try:
            cp_phase = cp.PropsSI("C", "P", p2, "T", T_out, fluid)
        except Exception:
            cp_phase = float("nan")
        try:
            h_phase = cp.PropsSI("H", "P", p2, "T", T_out, fluid)
        except Exception:
            h_phase = float("nan")

        rows.append(dict(
            phase=name,
            T=T_out, Tsat=T_sat,
            U=U, rho=rho, mu=mu, Re=Re,
            alpha=alpha_single,
            cp=cp_phase, MW=MW_val, h=h_phase,
        ))

    _print_table(
        "OUTLET PROPERTIES - Phases (T, Tsat, U, rho, mu, Re, alpha, Cp, MW, h)",
        cols, rows
    )

# ============================================================
#  MAIN MIXTURE RESULTS TABLE
# ============================================================

def print_main_results_table(mdot_pa: float,
                             fluid: str,
                             p2_bar: float,
                             D: float,
                             out: Dict[str, Any]) -> None:
    """
    Main table with:
    - phase-aware mdot
    - outlet phase (liquid/gas/two-phase)
    - T_out, T_sat(P2)
    - U_mix, rho_mix, mu_mix, Re_mix, Mach
    - sigma (surface tension) evaluated at T_sat(P2), Q=0 if two-phase.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    phase_out = out["phase_out"]

    # Surface tension:
    # - only meaningful if the outlet is two-phase (liquid–vapor interface)
    # - we use I(T_sat, Q=0) [N/m]; for single phase we return 0.0
    if phase_out == "two-phase":
        try:
            sigma = cp.PropsSI("I", "T", T_sat, "Q", 0, fluid)
        except Exception:
            sigma = float("nan")
    else:
        sigma = 0.0

    cols = [
        ("mdot [kg/s]",  "mdot",  12, ".6f"),
        ("Phase",        "phase", 10, "s"),
        ("T [K]",        "T",      8, ".3f"),
        ("Tsat [K]",     "Tsat",   9, ".3f"),
        ("U [m/s]",      "U",      9, ".3f"),
        ("rho [kg/m3]",  "rho",   12, ".3f"),
        ("mu [Pa·s]",    "mu",    12, ".3e"),
        ("Re [-]",       "Re",    12, ".3e"),
        ("Mach [-]",     "Mach",   9, ".3f"),
        ("sigma [N/m]",  "sigma", 12, ".3e"),
    ]

    rows = [dict(
        mdot  = mdot_pa,
        phase = phase_out,
        T     = out["T_out"],
        Tsat  = T_sat,
        U     = out["U_out"],
        rho   = out["rho_mix"],
        mu    = out["mu_mix"],
        Re    = out["Re_out"],
        Mach  = out["Mach"],
        sigma = sigma,
    )]

    _print_table(
        "MAIN RESULTS - Phase-aware mass flow and mixture properties",
        cols, rows
    )


def run_performance_case(fluid: str = "NitrousOxide",
                         p1_bar: float = 55.0,
                         p2_bar: float = 43.0,
                         T_line: float = 288.0,
                         D: float = 2.0e-3,
                         L: float = 10.0e-3,
                         Nh: int = 1,
                         Cd: float = 1.0,
                         Cd_mode: str = "user",
                         r_over_D: float = 0.05,
                         use_spi_comp: bool = True,
                         spi_n: float = None,
                         keep_area: bool = False) -> None:
    """
    Run a single performance case with the given parameters.

    D, L, r_over_D are per-hole geometric parameters.
    Nh is the number of identical holes.
    All mass flow rates reported in tables are TOTAL (sum over the Nh holes).
    Outlet properties are per hole.

    keep_area:
        True  -> nella tabella di input l'opzione "Keep A_tot const (Nh)" è ON
        False -> OFF.
    """
    Nh = max(int(Nh), 1)

    # ----- Cd determination -----
    if Cd_mode == "geom":
        Cd_geom, Re_ref = estimate_Cd_from_geometry_perf(
            fluid=fluid,
            p1_bar=p1_bar,
            p2_bar=p2_bar,
            T_line=T_line,
            D=D,
            L=L,
            r_over_D=r_over_D,
        )
        Cd = Cd_geom
        Cd_source = f"estimated from geometry (r/D={r_over_D:.3f}, Re≈{Re_ref:.2e})"
    else:
        Cd_source = "user-provided Cd"
        Re_ref = None  # optional

    # ---- 2) SPI / HEM / NHNE + phase-aware mass flow rates (per hole) ----
    mdot_spi_one = _mdot_spi(
        fluid, p1_bar, p2_bar, T_line, D, Cd,
        use_compress=use_spi_comp, n_isentropic=spi_n
    )
    mdot_hem_one = _mdot_hem(fluid, p1_bar, p2_bar, T_line, D, Cd)
    mdot_nhne_one, _ = _mdot_nhne(
        fluid, p1_bar, p2_bar, T_line, D, Cd,
        L_over_D=(L / D if D > 0.0 else None),
        K_RESIDENCE=0.0,
        use_spi_compress=use_spi_comp,
        spi_n=spi_n,
    )

    mdot_pa_one, model_used = compute_mdot_phaseaware(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        Cd=Cd,
        L=L,
        use_spi_compress=use_spi_comp,
        spi_n=spi_n,
        K_RESIDENCE=0.0,
    )

    # ---- Total mass flows (sum over Nh holes) ----
    mdot_spi  = Nh * mdot_spi_one
    mdot_hem  = Nh * mdot_hem_one
    mdot_nhne = Nh * mdot_nhne_one
    mdot_pa   = Nh * mdot_pa_one

    # ---- 3) Outlet state per hole, consistent with mdot_pa_one ----
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5

    out = nhne_out_state_from_mdot(
        fluid=fluid,
        p1=p1,
        p2=p2,
        T_line=T_line,
        D=D,
        mdot_nhne=mdot_pa_one,   # per-hole mdot
        h1_hint=None,
    )

    # ---- 4) Graphical interface: 4 tables in the same window ----
    show_output_tables_gui(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        L=L,
        r_over_D=r_over_D,
        Nh=Nh,
        Cd=Cd,
        Cd_mode=Cd_mode,
        Cd_source=Cd_source,
        spi_n=spi_n,
        use_spi_comp=use_spi_comp,
        out=out,
        mdot_spi=mdot_spi,
        mdot_hem=mdot_hem,
        mdot_nhne=mdot_nhne,
        mdot_pa=mdot_pa,
        model_used=model_used,
        keep_area=keep_area,
    )


def show_output_tables_gui(fluid: str,
                           p1_bar: float,
                           p2_bar: float,
                           T_line: float,
                           D: float,
                           L: float,
                           r_over_D: float,
                           Nh: int,
                           Cd: float,
                           Cd_mode: str,
                           Cd_source: str,
                           spi_n,
                           use_spi_comp: bool,
                           out: Dict[str, Any],
                           mdot_spi: float,
                           mdot_hem: float,
                           mdot_nhne: float,
                           mdot_pa: float,
                           model_used: str,
                           keep_area: bool = False) -> None:
    """
    Creates a single matplotlib window with four tables in a 2×2 layout:

      [0,0] INPUT PARAMETERS
      [0,1] OUTLET PROPERTIES
      [1,0] MASS FLOW TABLE
      [1,1] MAIN RESULTS

    The two tables on the left have the same width,
    and likewise for the two on the right.
    The tables on each row are aligned at the top.

    Nh = number of identical holes; mdot_* are TOTAL mass flow rates (sum over holes).
    D, L, r_over_D are per-hole geometric parameters.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping graphical output interface.")
        return

    p2 = p2_bar * 1e5
    phase_out = out["phase_out"]
    T_out = out["T_out"]

    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = T_out

    alpha_mix = float(out.get("alpha_out", 0.0))

    try:
        MW_val = cp.PropsSI("M", fluid) * 1000.0  # kg/kmol
    except Exception:
        MW_val = float("nan")

    # ========== 1) MASS FLOW TABLE (TOTAL mdot) ==========
    cols_mdot: List[Tuple[str, str]] = [
        ("Model",       "model"),
        ("mdot_total [kg/s]", "mdot"),
    ]
    rows_mdot = [
        {"model": "SPI",         "mdot": mdot_spi},
        {"model": "HEM",         "mdot": mdot_hem},
        {"model": "NHNE",        "mdot": mdot_nhne},
        {"model": "Phase-aware", "mdot": mdot_pa},
    ]

    # ========== 2) PHASE PROPERTIES (per hole) ==========
    cols_phase: List[Tuple[str, str]] = [
        ("Phase",         "phase"),
        ("T [K]",         "T"),
        ("T_sat [K]",     "Tsat"),
        ("U [m/s]",       "U"),
        ("rho [kg/m3]",   "rho"),
        ("mu [Pa·s]",     "mu"),
        ("Re [-]",        "Re"),
        ("alpha [-]",     "alpha"),
        ("Cp [J/kg/K]",   "cp"),
        ("MW [kg/kmol]",  "MW"),
        ("h [J/kg]",      "h"),
    ]
    rows_phase: List[Dict[str, Any]] = []

    if phase_out == "two-phase":
        rho_l = out["rho_l"]
        rho_v = out["rho_v"]
        j_l   = out["j_liq"]
        j_v   = out["j_gas"]

        mu_l  = _safe_viscosity(fluid, p2, T_out, phase="liq")
        mu_v  = _safe_viscosity(fluid, p2, T_out, phase="gas")

        Re_l  = rho_l * j_l * D / max(mu_l, 1e-12) if rho_l is not None else 0.0
        Re_v  = rho_v * j_v * D / max(mu_v, 1e-12) if rho_v is not None else 0.0

        try:
            cp_l = cp.PropsSI("C", "P", p2, "Q", 0, fluid)
        except Exception:
            cp_l = float("nan")
        try:
            cp_v = cp.PropsSI("C", "P", p2, "Q", 1, fluid)
        except Exception:
            cp_v = float("nan")
        try:
            h_l = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        except Exception:
            h_l = float("nan")
        try:
            h_v = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
        except Exception:
            h_v = float("nan")

        rows_phase.append(dict(
            phase="liquid",
            T=T_out, Tsat=T_sat,
            U=j_l, rho=rho_l, mu=mu_l, Re=Re_l,
            alpha=(1.0 - alpha_mix),
            cp=cp_l, MW=MW_val, h=h_l,
        ))
        rows_phase.append(dict(
            phase="vapor",
            T=T_out, Tsat=T_sat,
            U=j_v, rho=rho_v, mu=mu_v, Re=Re_v,
            alpha=alpha_mix,
            cp=cp_v, MW=MW_val, h=h_v,
        ))
    else:
        if phase_out == "gas":
            rho = out["rho_v"] if out["rho_v"] is not None else rho_singlephase_at_T(
                fluid, p2, T_out, side="gas"
            )
            mu   = _safe_viscosity(fluid, p2, T_out, phase="gas")
            name = "gas"
            alpha_single = 1.0
        else:
            rho = out["rho_l"] if out["rho_l"] is not None else rho_singlephase_at_T(
                fluid, p2, T_out, side="liq"
            )
            mu   = _safe_viscosity(fluid, p2, T_out, phase="liq")
            name = "liquid"
            alpha_single = 0.0

        U  = out["U_out"]
        Re = rho * U * D / max(mu, 1e-12)

        try:
            cp_phase = cp.PropsSI("C", "P", p2, "T", T_out, fluid)
        except Exception:
            cp_phase = float("nan")
        try:
            h_phase = cp.PropsSI("H", "P", p2, "T", T_out, fluid)
        except Exception:
            h_phase = float("nan")

        rows_phase.append(dict(
            phase=name,
            T=T_out, Tsat=T_sat,
            U=U, rho=rho, mu=mu, Re=Re,
            alpha=alpha_single,
            cp=cp_phase, MW=MW_val, h=h_phase,
        ))

    # ========== 3) MIXTURE RESULTS (TOTAL mdot) ==========
    if phase_out == "two-phase":
        try:
            sigma = cp.PropsSI("I", "T", T_sat, "Q", 0, fluid)
        except Exception:
            sigma = float("nan")
    else:
        sigma = 0.0

    cols_mix: List[Tuple[str, str]] = [
        ("mdot_total [kg/s]", "mdot"),
        ("Phase",             "phase"),
        ("T [K]",             "T"),
        ("T_sat [K]",         "Tsat"),
        ("U_mix [m/s]",       "U"),
        ("rho_mix [kg/m3]",   "rho"),
        ("mu_mix [Pa·s]",     "mu"),
        ("Re_mix [-]",        "Re"),
        ("Mach [-]",          "Mach"),
        ("sigma [N/m]",       "sigma"),
    ]
    rows_mix = [dict(
        mdot  = mdot_pa,
        phase = phase_out,
        T     = out["T_out"],
        Tsat  = T_sat,
        U     = out["U_out"],
        rho   = out["rho_mix"],
        mu    = out["mu_mix"],
        Re    = out["Re_out"],
        Mach  = out["Mach"],
        sigma = sigma,
    )]

    # ========== FIGURE WITH 4 TABLES (2×2 layout) ==========
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec
    import math as _math

    # --- INPUT table (vertical) ---
    if D > 0.0:
        L_over_D_val = L / D
        A_hole = 0.25 * math.pi * D * D
        Nh_eff = max(Nh, 1)
        A_tot = A_hole * Nh_eff
        r_val = r_over_D * D
        D_ref = D * math.sqrt(Nh_eff)
    else:
        L_over_D_val = float("nan")
        A_hole = float("nan")
        A_tot = float("nan")
        r_val = float("nan")
        D_ref = float("nan")

    # Se Cd_mode = "user", L e r non sono stati inseriti → mostriamo "-"
    if Cd_mode == "user":
        L_str = "-"
        r_str = "-"
        L_over_D_str = "-"
        r_over_D_str = "-"
    else:
        L_str = f"{L:.6f}"
        r_str = f"{r_val:.6f}"
        L_over_D_str = f"{L_over_D_val:.3f}"
        r_over_D_str = f"{r_over_D:.5f}"

    # Cd_source: se stimato dalla geometria, mostriamo il valore numerico
    if Cd_mode == "geom":
        Cd_source_display = f"{Cd:.4f}"
    else:
        Cd_source_display = Cd_source

    rows_input = [
        {"param": "Fluid",                "val": fluid},
        {"param": "T_line [K]",           "val": f"{T_line:.3f}"},
        {"param": "P1 [bar]",             "val": f"{p1_bar:.3f}"},
        {"param": "P2 [bar]",             "val": f"{p2_bar:.3f}"},
        {"param": "Orifice D_hole [m]",   "val": f"{D:.6f}"},
        {"param": "D_ref [m] (equiv.)",   "val": f"{D_ref:.6f}"},
        {"param": "Orifice L [m]",        "val": L_str},
        {"param": "L/D [-]",              "val": L_over_D_str},
        {"param": "Edge radius r [m]",    "val": r_str},
        {"param": "r/D [-]",              "val": r_over_D_str},
        {"param": "Nh [-]",               "val": f"{Nh:d}"},
        {"param": "A_h [m^2]",            "val": f"{A_hole:.8e}"},
        {"param": "A_tot [m^2]",          "val": f"{A_tot:.8e}"},
        {"param": "Cd [-]",               "val": f"{Cd:.4f}"},
        {"param": "Cd mode",              "val": Cd_mode},
        {"param": "Cd source",            "val": Cd_source_display},
        {"param": "Keep A_tot const (Nh)","val": "ON" if keep_area else "OFF"},
        {"param": "SPI compressible",     "val": "ON" if use_spi_comp else "OFF"},
        {"param": "n SPI",                "val": (f"{spi_n:.3f}" if spi_n is not None else "default")},
    ]

    cols_input: List[Tuple[str, str]] = [
        ("Parameter", "param"),
        ("Value",     "val"),
    ]

    # --- figure sizing ---
    n_rows_mdot  = len(rows_mdot) + 1
    n_rows_input = len(rows_input) + 1
    n_rows_phase = len(rows_phase) + 1
    n_rows_mix   = len(rows_mix) + 1

    fig_h = 0.20 * (max(n_rows_input, n_rows_phase) +
                    max(n_rows_mdot,  n_rows_mix)) + 1.0
    fig_w = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    top = max(n_rows_input, n_rows_phase)
    bottom = max(n_rows_mdot, n_rows_mix)

    gs = GridSpec(
        2, 2,
        height_ratios=[top * 1.15, bottom * 0.85],
        figure=fig,
    )

    ax_input = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_mdot  = fig.add_subplot(gs[1, 0])
    ax_mix   = fig.add_subplot(gs[1, 1])

    ax_input.axis("off")
    ax_phase.axis("off")
    ax_mdot.axis("off")
    ax_mix.axis("off")

    # ---------- INPUT Table ----------
    header_input = [c[0] for c in cols_input]
    data_input = [header_input] + [[r["param"], r["val"]] for r in rows_input]

    tab_input = ax_input.table(cellText=data_input,
                               loc="upper center", cellLoc="center")
    tab_input.auto_set_font_size(False)
    tab_input.set_fontsize(8)
    tab_input.scale(1.0, 1.1)
    for (row, col), cell in tab_input.get_celld().items():
        if row == 0:
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(weight="bold")

    # ---------- MASS FLOW Table ----------
    header_mdot = [c[0] for c in cols_mdot]
    data_mdot = [header_mdot]
    for r in rows_mdot:
        row_vals = []
        for (_, key) in cols_mdot:
            val = r.get(key, None)
            if isinstance(val, (int, float)) and val is not None:
                row_vals.append(f"{val:.6g}" if _math.isfinite(val) else "NaN")
            elif val is None:
                row_vals.append("-")
            else:
                row_vals.append(str(val))
        data_mdot.append(row_vals)

    tab_mdot = ax_mdot.table(cellText=data_mdot,
                             loc="upper center", cellLoc="center")
    tab_mdot.auto_set_font_size(False)
    tab_mdot.set_fontsize(8)
    tab_mdot.scale(1.0, 1.1)
    for (row, col), cell in tab_mdot.get_celld().items():
        if row == 0:
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(weight="bold")

    # ---------- OUTLET PROPERTIES Table (vertical, 3 columns) ----------
    prop_list = [
        ("T [K]",        "T"),
        ("T_sat [K]",    "Tsat"),
        ("U [m/s]",      "U"),
        ("rho [kg/m3]",  "rho"),
        ("mu [Pa·s]",    "mu"),
        ("Re [-]",       "Re"),
        ("alpha [-]",    "alpha"),
        ("Cp [J/kg/K]",  "cp"),
        ("MW [kg/kmol]", "MW"),
        ("h [J/kg]",     "h"),
    ]

    phase_names = []
    for name in ("liquid", "vapor", "gas"):
        if any(r["phase"] == name for r in rows_phase):
            phase_names.append(name)
    if not phase_names:
        phase_names = sorted({r["phase"] for r in rows_phase})

    header_phase_v = ["Phase"] + phase_names[:2]
    while len(header_phase_v) < 3:
        header_phase_v.append("")

    phase_dict = {r["phase"]: r for r in rows_phase}

    data_phase_v = [header_phase_v]
    for label, key in prop_list:
        row_vals = [label]
        for ph in phase_names[:2]:
            val = phase_dict.get(ph, {}).get(key, None)
            if isinstance(val, (int, float)) and val is not None:
                row_vals.append(f"{val:.6g}" if _math.isfinite(val) else "NaN")
            elif val is None:
                row_vals.append("-")
            else:
                row_vals.append(str(val))
        while len(row_vals) < 3:
            row_vals.append("")
        data_phase_v.append(row_vals)

    tab_phase = ax_phase.table(cellText=data_phase_v,
                               loc="upper center", cellLoc="center")
    tab_phase.auto_set_font_size(False)
    tab_phase.set_fontsize(7)
    tab_phase.scale(1.0, 1.2)
    for (row, col), cell in tab_phase.get_celld().items():
        if row == 0:
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(weight="bold")

    # ---------- MIXTURE RESULTS Table (vertical, 2 columns) ----------
    mix_row = rows_mix[0]
    prop_mix = [
        ("mdot_total [kg/s]", "mdot"),
        ("Phase",             "phase"),
        ("T [K]",             "T"),
        ("T_sat [K]",         "Tsat"),
        ("U_mix [m/s]",       "U"),
        ("rho_mix [kg/m3]",   "rho"),
        ("mu_mix [Pa·s]",     "mu"),
        ("Re_mix [-]",        "Re"),
        ("Mach [-]",          "Mach"),
        ("sigma [N/m]",       "sigma"),
    ]

    header_mix_v = ["Parameter", "Value"]
    data_mix_v = [header_mix_v]
    for label, key in prop_mix:
        val = mix_row.get(key, None)
        if isinstance(val, (int, float)) and val is not None:
            sval = f"{val:.6g}" if _math.isfinite(val) else "NaN"
        elif val is None:
            sval = "-"
        else:
            sval = str(val)
        data_mix_v.append([label, sval])

    tab_mix = ax_mix.table(cellText=data_mix_v,
                           loc="upper center", cellLoc="center")
    tab_mix.auto_set_font_size(False)
    tab_mix.set_fontsize(7)
    tab_mix.scale(1.0, 1.2)
    for (row, col), cell in tab_mix.get_celld().items():
        if row == 0:
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(weight="bold")

    # ---------- Align widths pairwise ----------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Left: input & mass flow
    bbox_in  = tab_input.get_window_extent(renderer)
    bbox_md  = tab_mdot.get_window_extent(renderer)
    if bbox_md.width > 0:
        scale_left = bbox_in.width / bbox_md.width
        tab_mdot.scale(scale_left, 1.0)

    # Right: phase properties & mixture
    bbox_ph  = tab_phase.get_window_extent(renderer)
    bbox_mix = tab_mix.get_window_extent(renderer)
    if bbox_mix.width > 0:
        scale_right = bbox_ph.width / bbox_mix.width
        tab_mix.scale(scale_right, 1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ---------- Titles ----------
    def attach_title(ax, table, text, fontsize=9):
        bbox = table.get_window_extent(renderer)
        bb_ax = bbox.transformed(ax.transAxes.inverted())
        y = bb_ax.y1 + 0.03
        ax.text(0.5, y, text, ha="center", va="bottom", fontsize=fontsize)

    attach_title(ax_input, tab_input, "INPUT PARAMETERS", fontsize=9)
    attach_title(
        ax_mdot,
        tab_mdot,
        "MASS FLOW TABLE – TOTAL mdot (SPI / HEM / NHNE / Phase-aware)\n"
        f"Model automatically selected (phase-aware): {model_used}",
        fontsize=8.5,
    )
    attach_title(ax_phase, tab_phase, "OUTLET PROPERTIES (per hole)", fontsize=9)
    attach_title(
        ax_mix,
        tab_mix,
        "MAIN RESULTS - TOTAL mass flow and mixture properties",
        fontsize=9,
    )

    fig.tight_layout(pad=1.0, h_pad=0.8)
    plt.show()

def launch_gui():
    """
    Tkinter window to set case parameters and launch
    run_performance_case() with a "Run performance" button.

    Inputs:
      - fluid, T_line, P1, P2
      - D (per hole), L, r, Nh
      - Cd, Cd mode (user / geom)
      - SPI compressible + n_SPI
      - D_ref: equivalent single-orifice diameter

    Logic:
      - Always: A_tot = (pi/4) * D_ref^2 = Nh * (pi/4) * D^2
      - If 'keep area' is checked:
            D_ref is the input, D is derived (readonly).
        else:
            D is the input, D_ref is derived (readonly).
    """
    root = tk.Tk()
    root.title("PyInjection – Injector performance")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")

    # =========================
    #  Input variables
    # =========================
    fluid_var    = tk.StringVar(value="NitrousOxide")
    Tline_var    = tk.StringVar(value="288.0")
    p1_var       = tk.StringVar(value="55.0")
    p2_var       = tk.StringVar(value="43.0")

    D_init       = 0.002   # initial per-hole diameter
    Nh_init      = 1       # initial number of holes
    D_ref_init   = D_init * math.sqrt(Nh_init)

    D_var        = tk.StringVar(value=f"{D_init:.6f}")       # per-hole diameter
    Dref_var     = tk.StringVar(value=f"{D_ref_init:.6f}")   # equivalent single orifice
    L_var        = tk.StringVar(value="0.010")
    r_var        = tk.StringVar(value="0.00010")
    rD_var       = tk.StringVar(value="0.05")
    Nh_var       = tk.StringVar(value=str(Nh_init))
    Cd_var       = tk.StringVar(value="1.0")
    spi_n_var    = tk.StringVar(value="1.2")
    Cd_mode_var  = tk.StringVar(value="user")
    spi_comp_var = tk.BooleanVar(value=True)
    keep_area_var = tk.BooleanVar(value=False)

    # =========================
    #  Derived variables (read-only)
    # =========================
    L_over_D_var   = tk.StringVar(value="5.000")
    A_h_var        = tk.StringVar(value="3.14159265e-06")  # per-hole area
    A_tot_var      = tk.StringVar(value="3.14159265e-06")  # total area
    Cd_source_var  = tk.StringVar(value="user-provided Cd")

    # Flags to avoid recursive updates
    updating_D     = {"flag": False}
    updating_Dref  = {"flag": False}
    updating_Nh    = {"flag": False}

        # =========================
    #  Derived quantities update
    # =========================
    def _update_derived():
        """Update L/D, r/D, A_h, A_tot based on current D, D_ref, Nh.

        This function is robust to '-' placeholders used for L and r
        when Cd_mode = 'user'.
        """
        # D: per-hole diameter
        try:
            D_val = float(D_var.get())
        except ValueError:
            D_val = 0.0

        # D_ref: equivalent single-orifice diameter
        try:
            Dref_val = float(Dref_var.get())
        except ValueError:
            Dref_val = 0.0

        # Nh: number of holes
        try:
            Nh_val = int(Nh_var.get())
        except ValueError:
            Nh_val = 1
        Nh_val = max(Nh_val, 1)

        # L and r can be "-" in user mode → treat as "not used"
        L_raw = L_var.get().strip()
        r_raw = r_var.get().strip()

        L_used = None if L_raw in ("", "-") else float(L_raw)
        r_used = None if r_raw in ("", "-") else float(r_raw)

        # ---- L/D and r/D ----
        if D_val > 0.0:
            if L_used is not None:
                L_over_D_var.set(f"{L_used / D_val:.3f}")
            else:
                # in user mode we show "-"
                L_over_D_var.set("-")

            if r_used is not None:
                rD_var.set(f"{r_used / D_val:.5f}")
            else:
                rD_var.set("-")

            A_h = 0.25 * math.pi * D_val * D_val
            A_h_var.set(f"{A_h:.8e}")
        else:
            # no valid D → areas not meaningful
            L_over_D_var.set("-" if L_used is None else "nan")
            rD_var.set("-" if r_used is None else "nan")
            A_h_var.set("nan")
            A_h = float("nan")

        # ---- Total area A_tot ----
        if Dref_val > 0.0:
            # D_ref defines the total area
            A_tot = 0.25 * math.pi * Dref_val * Dref_val
        elif D_val > 0.0:
            # fallback: from per-hole area and Nh
            A_tot = A_h * Nh_val
        else:
            A_tot = float("nan")

        if math.isfinite(A_tot):
            A_tot_var.set(f"{A_tot:.8e}")
        else:
            A_tot_var.set("nan")

    # =========================
    #  Coupled updates: D, D_ref, Nh
    # =========================
    def _on_D_changed(*_):
        if updating_D["flag"]:
            return
        # If keep_area is OFF, D is the primary input → update D_ref
        if not keep_area_var.get():
            try:
                D_val  = float(D_var.get())
                Nh_val = int(Nh_var.get())
            except ValueError:
                _update_derived()
                return
            Nh_val = max(Nh_val, 1)
            if D_val > 0.0:
                Dref_val = D_val * math.sqrt(Nh_val)
                updating_Dref["flag"] = True
                try:
                    Dref_var.set(f"{Dref_val:.6f}")
                finally:
                    updating_Dref["flag"] = False
        _update_derived()

    def _on_Dref_changed(*_):
        if updating_Dref["flag"]:
            return
        # If keep_area is ON, D_ref is the primary input → update D
        if keep_area_var.get():
            try:
                Dref_val = float(Dref_var.get())
                Nh_val   = int(Nh_var.get())
            except ValueError:
                _update_derived()
                return
            Nh_val = max(Nh_val, 1)
            if Nh_val > 0 and Dref_val > 0.0:
                D_val = Dref_val / math.sqrt(Nh_val)
                updating_D["flag"] = True
                try:
                    D_var.set(f"{D_val:.6f}")
                finally:
                    updating_D["flag"] = False
        _update_derived()

    def _on_Nh_changed(*_):
        if updating_Nh["flag"]:
            return
        updating_Nh["flag"] = True
        try:
            try:
                Nh_val = int(Nh_var.get())
            except ValueError:
                Nh_val = 1
            Nh_val = max(Nh_val, 1)
            Nh_var.set(str(Nh_val))

            if keep_area_var.get():
                # Area fixed by D_ref → update D
                try:
                    Dref_val = float(Dref_var.get())
                except ValueError:
                    Dref_val = 0.0
                if Nh_val > 0 and Dref_val > 0.0:
                    D_val = Dref_val / math.sqrt(Nh_val)
                    updating_D["flag"] = True
                    try:
                        D_var.set(f"{D_val:.6f}")
                    finally:
                        updating_D["flag"] = False
            else:
                # Per-hole D is primary → update D_ref
                try:
                    D_val = float(D_var.get())
                except ValueError:
                    D_val = 0.0
                if D_val > 0.0:
                    Dref_val = D_val * math.sqrt(Nh_val)
                    updating_Dref["flag"] = True
                    try:
                        Dref_var.set(f"{Dref_val:.6f}")
                    finally:
                        updating_Dref["flag"] = False
        finally:
            updating_Nh["flag"] = False

        _update_derived()

    # Traces
    D_var.trace_add("write", _on_D_changed)
    Dref_var.trace_add("write", _on_Dref_changed)
    Nh_var.trace_add("write", _on_Nh_changed)
    L_var.trace_add("write", lambda *_: _update_derived())
    r_var.trace_add("write", lambda *_: _update_derived())

    # =========================
    #  Checkbox behavior (keep area)
    # =========================
    def _apply_mode_states():
        """Enable/disable entries according to keep_area_var."""
        if keep_area_var.get():
            d_entry.config(state="readonly")
            dref_entry.config(state="normal")
        else:
            d_entry.config(state="normal")
            dref_entry.config(state="readonly")

    def _on_keep_area_toggle():
        # Preserve total area while switching mode
        try:
            Nh_val   = int(Nh_var.get())
            D_val    = float(D_var.get())
            Dref_val = float(Dref_var.get())
        except ValueError:
            Nh_val   = max(int(Nh_var.get() or "1"), 1)
            D_val    = float(D_var.get() or "0.0")
            Dref_val = float(Dref_var.get() or "0.0")

        Nh_val = max(Nh_val, 1)
        Nh_var.set(str(Nh_val))

        if keep_area_var.get():
            # entering keep-area mode: D_ref must represent current A_tot
            if D_val > 0.0:
                Dref_val = D_val * math.sqrt(Nh_val)
                updating_Dref["flag"] = True
                try:
                    Dref_var.set(f"{Dref_val:.6f}")
                finally:
                    updating_Dref["flag"] = False
            # then D will be kept consistent via _on_Dref_changed
            _on_Dref_changed()
        else:
            # exiting keep-area mode: D (per hole) is primary
            _on_D_changed()

        _apply_mode_states()

    # =========================
    #  Layout
    # =========================
    row = 0
    ttk.Label(frame, text="Parameters", font=("Segoe UI", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    row += 1

    def add_row(label_text, var, editable=True):
        nonlocal row
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
        if editable:
            entry = ttk.Entry(frame, textvariable=var, width=18)
        else:
            entry = ttk.Entry(frame, textvariable=var, width=18, state="readonly")
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        return entry

    # real inputs
    fluid_entry   = add_row("Fluid",       fluid_var)
    Tline_entry   = add_row("T_line [K]",  Tline_var)
    p1_entry      = add_row("P1 [bar]",    p1_var)
    p2_entry      = add_row("P2 [bar]",    p2_var)

    d_entry    = add_row("D [m] (per hole)", D_var)              # state changes with checkbox
    dref_entry = add_row("D_ref [m] (equiv. single)", Dref_var)  # idem
    L_entry    = add_row("L [m]", L_var)

    # derived
    L_over_D_entry = add_row("L/D [-]",            L_over_D_var, editable=False)
    r_entry        = add_row("r [m]",             r_var)
    rD_entry       = add_row("r/D [-]",           rD_var,        editable=False)
    Nh_entry       = add_row("Nh [-]",            Nh_var)
    A_h_entry      = add_row("A_h [m^2]",         A_h_var,       editable=False)
    A_tot_entry    = add_row("A_tot [m^2]",       A_tot_var,     editable=False)

    # Cd & SPI
    cd_entry       = add_row("Cd [-]", Cd_var)
    spi_n_entry    = add_row("n SPI (opt.)", spi_n_var)

    ttk.Label(frame, text="Cd mode").grid(row=row, column=0, sticky="w", pady=(4, 2))
    cd_option = ttk.Combobox(
        frame,
        textvariable=Cd_mode_var,
        values=["user", "geom"],
        state="readonly",
        width=15,
    )
    cd_option.grid(row=row, column=1, sticky="w", pady=(4, 2))

    # ======= Gestione Cd_mode: abilita/disabilita L e r =======
    def _update_cd_entry_state(*_):
        # Cd: editabile solo in modalità "user"
        if Cd_mode_var.get() == "geom":
            cd_entry.config(state="readonly")
        else:
            cd_entry.config(state="normal")

        # L e r: usati solo per Cd geom → in "user" si oscurano e mostrano "-"
        if Cd_mode_var.get() == "geom":
            # torna a numerico, editabile
            if L_var.get().strip() == "-":
                L_var.set("0.010")
            if r_var.get().strip() == "-":
                r_var.set("0.00010")
            L_entry.config(state="normal")
            r_entry.config(state="normal")
        else:
            # modalità user: L e r non usati → li oscuriamo e li mettiamo a "-"
            L_entry.config(state="readonly")
            r_entry.config(state="readonly")
            L_var.set("-")
            r_var.set("-")
            L_over_D_var.set("-")
            rD_var.set("-")

    Cd_mode_var.trace_add("write", _update_cd_entry_state)
    _update_cd_entry_state()
    row += 1

    cd_source_entry = add_row("Cd source", Cd_source_var, editable=False)

    # keep area checkbox
    chk_area = ttk.Checkbutton(
        frame,
        text="Keep total area A_tot constant when Nh changes",
        variable=keep_area_var,
        command=_on_keep_area_toggle,
    )
    chk_area.grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 4))
    row += 1

    # SPI checkbox
    def _on_spi_comp_toggle():
        if spi_comp_var.get():
            spi_n_entry.config(state="normal")
            if not spi_n_var.get().strip():
                spi_n_var.set("1.2")
        else:
            spi_n_entry.config(state="disabled")

    chk_spi = ttk.Checkbutton(
        frame,
        text="SPI compressible",
        variable=spi_comp_var,
        command=_on_spi_comp_toggle,
    )
    chk_spi.grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 6))
    row += 1

    _on_spi_comp_toggle()
    _apply_mode_states()
    _update_derived()

    # ========= Tooltips for all main inputs =========
    ToolTip(
        fluid_entry,
        "CoolProp fluid name used for property evaluation.\n"
        "For this tool you typically use 'NitrousOxide'."
    )
    ToolTip(
        Tline_entry,
        "Feeding-line bulk temperature [K].\n"
        "Used to evaluate thermophysical properties and phase behaviour."
    )
    ToolTip(
        p1_entry,
        "Upstream/inlet pressure P1 [bar].\n"
        "This is the pressure before the injector orifice."
    )
    ToolTip(
        p2_entry,
        "Back pressure P2 [bar].\n"
        "This is the pressure at the injector outlet / chamber side."
    )
    ToolTip(
        d_entry,
        "diameter of each single holeof the injector D [m].\n"
        "Also used to compute L/D and r/D."
    )
    ToolTip(
        dref_entry,
        "Equivalent of single-orifice diameter D_ref [m].\n"
        "When 'Keep total area' is checked, D_ref is the primary input."
    )
    ToolTip(
        L_entry,
        "Orifice length L [m] (per hole).\n"
        "Used to build the aspect ratio L/D.\n"
        "Relevant only when Cd mode is 'geom'."
    )
    ToolTip(
        L_over_D_entry,
        "Aspect ratio L/D of the orifice,\n"
        "computed from the current values of L and D."
    )
    ToolTip(
        r_entry,
        "Inlet edge radius r [m] (per hole).\n"
        "Used in the geometric Cd model via the ratio r/D."
    )
    ToolTip(
        rD_entry,
        "Edge-radius ratio r/D.\n"
        "This is one of the inputs used by the geometric Cd model."
    )
    ToolTip(
        Nh_entry,
        "Number of identical injector holes Nh.\n"
        "If D is the primary input, total area is A_tot = Nh * A_h.\n"
        "If D_ref is primary (keep area ON), D is computed from D_ref and Nh."
    )
    ToolTip(
        A_h_entry,
        "Single-hole area A_h [m^2].\n"
        "Computed as A_h = (pi/4) * D^2."
    )
    ToolTip(
        A_tot_entry,
        "Total injector area A_tot [m^2].\n"
        "Computed as A_tot = Nh * A_h = (pi/4) * D_ref^2."
    )
    ToolTip(
        cd_entry,
        "Discharge coefficient Cd used in all mass-flow models.\n"
        "In 'user' mode this is taken directly from the input.\n"
        "In 'geom' mode it is overridden by the value estimated from geometry."
    )
    ToolTip(
        spi_n_entry,
        "Isentropic exponent n used in the SPI compressible correction.\n"
        "For N2O a typical value is n ≈ 1.2."
    )
    ToolTip(
        cd_option,
        "Cd mode:\n"
        "  - 'user': use the input Cd value as-is.\n"
        "  - 'geom': estimate Cd from L/D, r/D and Reynolds number."
    )
    ToolTip(
        cd_source_entry,
        "Text description of how Cd has been obtained\n"
        "(user-provided or estimated from geometry, with reference Re)."
    )
    ToolTip(
        chk_spi,
        "If checked, the SPI model uses a compressible correction\n"
        "with exponent n (n SPI). If unchecked, it uses the incompressible version."
    )
    ToolTip(
        chk_area,
        "When checked, D_ref defines the total area A_tot.\n"
        "D (per hole) is computed as D_ref / sqrt(Nh).\n"
        "When unchecked, D is the primary input and D_ref follows."
    )

    # =========================
    #  Run callback
    # =========================
    def on_run():
        try:
            fluid  = fluid_var.get().strip() or "NitrousOxide"
            T_line = float(Tline_var.get())
            p1_bar = float(p1_var.get())
            p2_bar = float(p2_var.get())
            D      = float(D_var.get())
            Nh_val = int(Nh_var.get())
            Nh_val = max(Nh_val, 1)

            Cd_mode = Cd_mode_var.get()

            # L e r: SOLO se geom, altrimenti non li leggiamo (sono "-")
            if Cd_mode == "geom":
                L = float(L_var.get())
                r = float(r_var.get())
                r_over_D = r / D if D > 0.0 else float("nan")
                rD_var.set(f"{r_over_D:.5f}")
            else:
                # in modalità user non servono alla fisica del Cd geom
                L = 0.0
                r = 0.0
                r_over_D = float("nan")
                rD_var.set("-")

            Cd_in  = float(Cd_var.get())

            spi_n_str = spi_n_var.get().strip()
            if spi_n_str:
                try:
                    spi_n = float(spi_n_str)
                except ValueError:
                    spi_n = 1.2
                    spi_n_var.set("1.2")
            else:
                spi_n = None

            use_spi_comp = bool(spi_comp_var.get())
            keep_area_flag = bool(keep_area_var.get())
        except ValueError as e:
            messagebox.showerror("Input error", f"Numeric conversion error:\n{e}")
            return

        if Cd_mode == "geom":
            try:
                Cd_geom, Re_ref = estimate_Cd_from_geometry_perf(
                    fluid=fluid,
                    p1_bar=p1_bar,
                    p2_bar=p2_bar,
                    T_line=T_line,
                    D=D,
                    L=L,
                    r_over_D=r_over_D,
                )
                Cd = Cd_geom
                Cd_source_str = (
                    f"estimated from geometry (r/D={r_over_D:.3f}, Re≈{Re_ref:.2e})"
                )
            except Exception as exc:
                messagebox.showerror(
                    "Geometric Cd estimation error",
                    f"Unable to estimate Cd from geometry:\n{exc}",
                )
                return
        else:
            Cd = Cd_in
            Cd_source_str = "user-provided Cd"

        Cd_source_var.set(Cd_source_str)
        Cd_var.set(f"{Cd:.4f}")

        # L e r sono significativi solo in modalità geom; in user li abbiamo
        # già posti a 0.0 / nan e non vengono usati dalla fisica.
        run_performance_case(
            fluid=fluid,
            p1_bar=p1_bar,
            p2_bar=p2_bar,
            T_line=T_line,
            D=D,
            L=L,
            Nh=Nh_val,
            Cd=Cd,
            Cd_mode=Cd_mode,
            r_over_D=r_over_D,
            use_spi_comp=use_spi_comp,
            spi_n=spi_n,
            keep_area=keep_area_flag,
        )

    btn = ttk.Button(frame, text="Run performance", command=on_run)
    btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    frame.columnconfigure(1, weight=1)

    root.mainloop()

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-case N2O multi-orifice injector (phase-aware backend + tables)."
    )
    parser.add_argument("--p1", type=float, default=55.0,
                        help="P1 inlet pressure [bar] (default: 55)")
    parser.add_argument("--p2", type=float, default=43.0,
                        help="P2 outlet/back pressure [bar] (default: 43)")
    parser.add_argument("--Tline", type=float, default=288.0,
                        help="Feeding line temperature [K] (default: 288)")
    parser.add_argument("--D", type=float, default=2.0e-3,
                        help="Orifice diameter [m] (default: 2.0e-3)")
    parser.add_argument("--L", type=float, default=10.0e-3,
                        help="Orifice length [m] (default: 10.0e-3)")
    parser.add_argument("--Cd", type=float, default=1,
                        help="Discharge coefficient [-] (default: 1)")
    parser.add_argument("--Cd-mode", type=str, default="user",
                        choices=["user", "geom"],
                        help="Cd determination mode: "
                             "'user' = use --Cd, 'geom' = estimate from geometry.")
    parser.add_argument("--rD", type=float, default=0.05,
                        help="Edge radius ratio r/D for geometric Cd estimation "
                             "(used only if --Cd-mode=geom, but always reported in tables).")
    parser.add_argument("--fluid", type=str, default="NitrousOxide",
                        help="Fluid name for CoolProp (default: NitrousOxide)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable SPI compressibility correction")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Isentropic exponent n for SPI (optional, e.g. 1.2)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch graphical input interface")
    parser.add_argument("--Nh", type=int, default=1,
                        help="Number of identical orifices (holes) (default: 1)")
    parser.add_argument("--n-holes", dest="Nh", type=int, default=1,
                        help="Alias: number of identical orifices (same as --Nh)")


    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    use_spi_comp = not args.no_compress

    run_performance_case(
        fluid=args.fluid,
        p1_bar=float(args.p1),
        p2_bar=float(args.p2),
        T_line=float(args.Tline),
        D=float(args.D),
        L=float(args.L),
        Nh=int(args.Nh),
        Cd=float(args.Cd),
        Cd_mode=args.Cd_mode,
        r_over_D=float(args.rD),
        use_spi_comp=use_spi_comp,
        spi_n=args.spi_n,
    )



if __name__ == "__main__":
    # If there are no command line arguments,
    # force GUI mode by adding "--gui" to sys.argv
    if len(sys.argv) == 1:
        sys.argv.append("--gui")
    main()
