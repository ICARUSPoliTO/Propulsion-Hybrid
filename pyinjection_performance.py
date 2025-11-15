"""
PyInjection_0D_V5.py
--------------------

Single-case plain-orifice N2O injector:
- usa il backend fisico/termodinamico in pyinjection_core.py
- stampa tabelle di input, portate e proprietà in uscita.

Questo file NON contiene più la fisica dei modelli; è solo un wrapper
per la CLI e la formattazione dell'output.
"""

import math
import argparse
from typing import Dict, Any, List, Tuple

import CoolProp.CoolProp as cp

# ============================================================
#  IMPORT DAL CORE
# ============================================================

from pyinjection_core import (
    _mdot_spi,
    _mdot_hem,
    _mdot_nhne,
    compute_mdot_phaseaware,
    nhne_out_state_from_mdot,
    _safe_viscosity,
    rho_singlephase_at_T,
)

# ============================================================
#  UTILITÀ DI STAMPA TABELLE
# ============================================================

def _print_table(title: str,
                 columns: List[Tuple[str, str, int, str]],
                 rows: List[Dict[str, Any]]) -> None:
    """
    columns: list of (header, key, width, fmt)
             fmt è solo per i numeri (es. '.3f', '.3e'); per stringhe è ignorato.
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
#  TABELLA INPUT
# ============================================================

def print_inputs_table(params: Dict[str, Any]) -> None:
    cols = [
        ("Parametro", "k", 24, "s"),
        ("Valore",    "v", 24, "s"),
    ]
    D = params["D"]
    A = 0.25 * math.pi * D**2

    rows = [
        {"k": "Fluido",           "v": params["fluid"]},
        {"k": "T_line [K]",       "v": f'{params["T_line"]:.3f}'},
        {"k": "P1 [bar]",         "v": f'{params["p1_bar"]:.3f}'},
        {"k": "P2 [bar]",         "v": f'{params["p2_bar"]:.3f}'},
        {"k": "D orifizio [m]",   "v": f'{D:.6f}'},
        {"k": "L orifizio [m]",   "v": f'{params["L"]:.6f}'},
        {"k": "A (da D) [m^2]",   "v": f'{A:.8e}'},
        {"k": "Cd [-]",           "v": f'{params["Cd"]:.4f}'},
    ]
    _print_table("TABELLA INPUT – Parametri del caso", cols, rows)


# ============================================================
#  TABELLA PORTATE
# ============================================================

def print_mdot_table(mdot_spi: float,
                     mdot_hem: float,
                     mdot_nhne: float,
                     mdot_pa: float,
                     model_used: str) -> None:
    cols = [
        ("Modello",       "model",     16, "s"),
        ("mdot [kg/s]",   "mdot",      18, ".6f"),
    ]
    rows = [
        {"model": "SPI",          "mdot": mdot_spi},
        {"model": "HEM",          "mdot": mdot_hem},
        {"model": "NHNE",         "mdot": mdot_nhne},
        {"model": "Phase-aware",  "mdot": mdot_pa},
    ]
    _print_table("TABELLA PORTATE – SPI / HEM / NHNE / Phase-aware", cols, rows)
    print(f"Modello selezionato automaticamente (phase-aware): {model_used}\n")


# ============================================================
#  TABELLA PROPRIETÀ DI FASE IN USCITA
# ============================================================

def print_phase_properties_table(fluid: str,
                                 p2_bar: float,
                                 D: float,
                                 out: Dict[str, Any]) -> None:
    """
    Proprietà in uscita delle singole fasi:
    - se bifase: liquido + vapore (stessa T_out, T_sat, ma j_liq/j_gas, ρ, μ, Re)
    - se monofase: unica riga per la fase presente.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    phase = out["phase_out"]
    T_out = out["T_out"]

    cols = [
        ("Fase",        "phase",    12, "s"),
        ("T [K]",       "T",        12, ".3f"),
        ("T_sat(P2)",   "Tsat",     12, ".3f"),
        ("U [m/s]",     "U",        12, ".3f"),
        ("rho [kg/m3]", "rho",      14, ".3f"),
        ("mu [Pa·s]",   "mu",       14, ".3e"),
        ("Re [-]",      "Re",       14, ".3e"),
    ]
    rows: List[Dict[str, Any]] = []

    if phase == "two-phase":
        rho_l = out["rho_l"]
        rho_v = out["rho_v"]
        j_l   = out["j_liq"]
        j_v   = out["j_gas"]

        mu_l  = _safe_viscosity(fluid, p2, phase="liq")
        mu_v  = _safe_viscosity(fluid, p2, T_out, phase="gas")

        Re_l  = rho_l * j_l * D / max(mu_l, 1e-12) if rho_l is not None else 0.0
        Re_v  = rho_v * j_v * D / max(mu_v, 1e-12) if rho_v is not None else 0.0

        rows.append(dict(
            phase="liquido",
            T=T_out, Tsat=T_sat,
            U=j_l, rho=rho_l, mu=mu_l, Re=Re_l,
        ))
        rows.append(dict(
            phase="vapore",
            T=T_out, Tsat=T_sat,
            U=j_v, rho=rho_v, mu=mu_v, Re=Re_v,
        ))

    else:
        # singola fase
        if phase == "gas":
            rho = out["rho_v"] if out["rho_v"] is not None else rho_singlephase_at_T(fluid, p2, T_out, side="gas")
            mu  = _safe_viscosity(fluid, p2, T_out, phase="gas")
            name = "gas"
        else:
            rho = out["rho_l"] if out["rho_l"] is not None else rho_singlephase_at_T(fluid, p2, T_out, side="liq")
            mu  = _safe_viscosity(fluid, p2, T_out, phase="liq")
            name = "liquido"

        U = out["U_out"]
        Re = rho * U * D / max(mu, 1e-12)

        rows.append(dict(
            phase=name,
            T=T_out, Tsat=T_sat,
            U=U, rho=rho, mu=mu, Re=Re,
        ))

    _print_table("PROPRIETÀ DI USCITA – Fasi (T, Tsat, U, rho, mu, Re)", cols, rows)


# ============================================================
#  TABELLA RISULTATI PRINCIPALI MISCELA
# ============================================================

def print_main_results_table(mdot_pa: float,
                             fluid: str,
                             p2_bar: float,
                             D: float,
                             out: Dict[str, Any]) -> None:
    """
    Tabella principale con:
    - mdot phase-aware
    - fase di uscita (liquid/gas/two-phase)
    - T_out, T_sat(P2)
    - U_mix, rho_mix, mu_mix, Re_mix, Mach.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    cols = [
        ("mdot_PA [kg/s]",   "mdot",      16, ".6f"),
        ("fase uscita",      "phase",     14, "s"),
        ("T_out [K]",        "T",         12, ".3f"),
        ("T_sat(P2) [K]",    "Tsat",      13, ".3f"),
        ("U_mix [m/s]",      "U",         12, ".3f"),
        ("rho_mix [kg/m3]",  "rho",       16, ".3f"),
        ("mu_mix [Pa·s]",    "mu",        16, ".3e"),
        ("Re_mix [-]",       "Re",        16, ".3e"),
        ("Mach [-]",         "Mach",      12, ".3f"),
    ]
    rows = [dict(
        mdot = mdot_pa,
        phase= out["phase_out"],
        T    = out["T_out"],
        Tsat = T_sat,
        U    = out["U_out"],
        rho  = out["rho_mix"],
        mu   = out["mu_mix"],
        Re   = out["Re_out"],
        Mach = out["Mach"],
    )]
    _print_table("RISULTATI PRINCIPALI – Portata phase-aware e proprietà miscela", cols, rows)


# ============================================================
#  MAIN (CLI SINGLE CASE)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-case N2O plain-orifice injector (phase-aware backend + tables)."
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
    parser.add_argument("--Cd", type=float, default=0.9875,
                        help="Discharge coefficient [-] (default: 0.9875)")
    parser.add_argument("--fluid", type=str, default="NitrousOxide",
                        help="Fluid name for CoolProp (default: NitrousOxide)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable SPI compressibility correction")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Isentropic exponent n for SPI (optional, e.g. 1.2)")

    args = parser.parse_args()

    fluid   = args.fluid
    p1_bar  = float(args.p1)
    p2_bar  = float(args.p2)
    T_line  = float(args.Tline)
    D       = float(args.D)
    L       = float(args.L)
    Cd      = float(args.Cd)
    use_spi_compress = not args.no_compress
    spi_n   = args.spi_n

    # ---- 1) Tabella input ----
    inputs = dict(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        L=L,
        Cd=Cd,
    )
    print_inputs_table(inputs)

    # ---- 2) Portate SPI / HEM / NHNE + phase-aware ----
    mdot_spi = _mdot_spi(fluid, p1_bar, p2_bar, T_line, D, Cd,
                         use_compress=use_spi_compress, n_isentropic=spi_n)
    mdot_hem = _mdot_hem(fluid, p1_bar, p2_bar, T_line, D, Cd)
    mdot_nhne, _ = _mdot_nhne(fluid, p1_bar, p2_bar, T_line, D, Cd,
                              L_over_D=(L / D if D > 0.0 else None),
                              K_RESIDENCE=0.0,
                              use_spi_compress=use_spi_compress,
                              spi_n=spi_n)

    mdot_pa, model_used = compute_mdot_phaseaware(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        Cd=Cd,
        L=L,
        use_spi_compress=use_spi_compress,
        spi_n=spi_n,
        K_RESIDENCE=0.0,
    )

    print_mdot_table(mdot_spi, mdot_hem, mdot_nhne, mdot_pa, model_used)

    # ---- 3) Stato di uscita coerente con mdot phase-aware ----
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5

    # se hai già h1 dal core puoi passarlo, altrimenti lasciamo h1_hint=None
    out = nhne_out_state_from_mdot(
        fluid=fluid,
        p1=p1,
        p2=p2,
        T_line=T_line,
        D=D,
        mdot_nhne=mdot_pa,
        h1_hint=None,
    )

    # Tabella proprietà di fase (singole fasi o bifase)
    print_phase_properties_table(fluid, p2_bar, D, out)

    # ---- 4) Tabella risultati principali della miscela ----
    print_main_results_table(mdot_pa, fluid, p2_bar, D, out)


if __name__ == "__main__":
    main()
