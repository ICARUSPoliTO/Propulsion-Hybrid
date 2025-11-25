"""
pyinjection_performance.py
--------------------

Single-case plain-orifice N2O injector:
- usa il backend fisico/termodinamico in pyinjection_core.py
- può usare un Cd imposto dall'utente oppure stimato dalla geometria
  tramite il modello geometrico del core
- stampa tabelle di input, portate e proprietà in uscita.

Questo file non contiene la fisica dei modelli; è solo un wrapper
per la formattazione dell'output.
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
    estimate_Cd_from_geometry,  # Cd geometrico (usa già rho_singlephase_at_T, _safe_viscosity, ecc.)
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
    """
    Stampa la tabella degli input usando:
    - fluid, T_line, P1, P2
    - geometria: D, L, L/D, r/D, r, area A
    - Cd, modalità e sorgente (user / geometrico).
    """
    cols = [
        ("Parametro", "k", 24, "s"),
        ("Valore",    "v", 24, "s"),
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
        {"k": "Fluido",           "v": params["fluid"]},
        {"k": "T_line [K]",       "v": f'{params["T_line"]:.3f}'},
        {"k": "P1 [bar]",         "v": f'{params["p1_bar"]:.3f}'},
        {"k": "P2 [bar]",         "v": f'{params["p2_bar"]:.3f}'},
        {"k": "D orifizio [m]",   "v": f'{D:.6f}'},
        {"k": "L orifizio [m]",   "v": f'{L:.6f}'},
        {"k": "L/D [-]",          "v": f'{L_over_D:.3f}'},
    ]

    # Geometria di raccordo se disponibile
    if r_over_D is not None:
        rows.append({"k": "r/D [-]",        "v": f'{r_over_D:.4f}'})
        rows.append({"k": "r raccordo [m]", "v": f'{r_edge:.6f}'})

    rows.extend([
        {"k": "Area foro A [m^2]", "v": f'{A:.8e}'},
        {"k": "Cd [-]",            "v": f'{params["Cd"]:.4f}'},
    ])

    # modalità e origine del Cd (user / geom)
    if "Cd_mode" in params:
        rows.append({"k": "Modalità Cd", "v": params["Cd_mode"]})
    if "Cd_source" in params:
        rows.append({"k": "Origine Cd", "v": params["Cd_source"]})

    _print_table("TABELLA INPUT - Parametri del caso", cols, rows)


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
    - se bifase: liquido + vapore (stessa T_out, T_sat, ma j_liq/j_gas, rho, mu, Re)
    - se monofase: unica riga per la fase presente.

    Per densità e viscosità si usano le utility del core
    (rho_singlephase_at_T, _safe_viscosity); CoolProp è usato solo
    per T_sat, Cp, k, h.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    phase = out["phase_out"]
    T_out = out["T_out"]

    cols = [
        ("Fase",         "phase",   10, "s"),
        ("T [K]",        "T",        8, ".3f"),
        ("T_sat(P2)",    "Tsat",    10, ".3f"),
        ("U [m/s]",      "U",       10, ".3f"),
        ("rho [kg/m3]",  "rho",     12, ".3f"),
        ("mu [Pa·s]",    "mu",      12, ".3e"),
        ("Re [-]",       "Re",      12, ".3e"),
        ("alpha [-]",    "alpha",    8, ".3f"),
        ("Cp [J/kg/K]",  "cp",      12, ".3e"),
        ("k [W/m/K]",    "k",       12, ".3e"),
        ("MW [kg/kmol]", "MW",      12, ".3f"),
        ("h [J/kg]",     "h",       14, ".3e"),
    ]
    rows: List[Dict[str, Any]] = []

    # volume fraction complessiva della fase vapore
    alpha_mix = float(out.get("alpha_out", 0.0))

    # MW è lo stesso per entrambe le fasi
    try:
        MW_val = cp.PropsSI("M", fluid) * 1000.0  # kg/kmol
    except Exception:
        MW_val = float("nan")

    if phase == "two-phase":
        rho_l = out["rho_l"]
        rho_v = out["rho_v"]
        j_l   = out["j_liq"]
        j_v   = out["j_gas"]

        # viscosità da core (con T_out)
        mu_l  = _safe_viscosity(fluid, p2, T_out, phase="liq")
        mu_v  = _safe_viscosity(fluid, p2, T_out, phase="gas")

        Re_l  = rho_l * j_l * D / max(mu_l, 1e-12) if rho_l is not None else 0.0
        Re_v  = rho_v * j_v * D / max(mu_v, 1e-12) if rho_v is not None else 0.0

        # Proprietà termiche alle condizioni sature lato liquido / vapore
        try:
            cp_l = cp.PropsSI("C", "P", p2, "Q", 0, fluid)
        except Exception:
            cp_l = float("nan")
        try:
            cp_v = cp.PropsSI("C", "P", p2, "Q", 1, fluid)
        except Exception:
            cp_v = float("nan")

        try:
            k_l = cp.PropsSI("L", "P", p2, "Q", 0, fluid)
        except Exception:
            k_l = float("nan")
        try:
            k_v = cp.PropsSI("L", "P", p2, "Q", 1, fluid)
        except Exception:
            k_v = float("nan")

        try:
            h_l = cp.PropsSI("H", "P", p2, "Q", 0, fluid)
        except Exception:
            h_l = float("nan")
        try:
            h_v = cp.PropsSI("H", "P", p2, "Q", 1, fluid)
        except Exception:
            h_v = float("nan")

        # Frazione volumetrica liquido = 1 - alpha_mix
        rows.append(dict(
            phase="liquido",
            T=T_out, Tsat=T_sat,
            U=j_l, rho=rho_l, mu=mu_l, Re=Re_l,
            alpha=(1.0 - alpha_mix),
            cp=cp_l, k=k_l, MW=MW_val, h=h_l,
        ))
        # Frazione volumetrica vapore = alpha_mix
        rows.append(dict(
            phase="vapore",
            T=T_out, Tsat=T_sat,
            U=j_v, rho=rho_v, mu=mu_v, Re=Re_v,
            alpha=alpha_mix,
            cp=cp_v, k=k_v, MW=MW_val, h=h_v,
        ))

    else:
        # monofase: calcolo proprietà alla condizione (P2, T_out)
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
            name = "liquido"
            alpha_single = 0.0

        U  = out["U_out"]
        Re = rho * U * D / max(mu, 1e-12)

        try:
            cp_phase = cp.PropsSI("C", "P", p2, "T", T_out, fluid)
        except Exception:
            cp_phase = float("nan")

        try:
            k_phase = cp.PropsSI("L", "P", p2, "T", T_out, fluid)
        except Exception:
            k_phase = float("nan")

        try:
            h_phase = cp.PropsSI("H", "P", p2, "T", T_out, fluid)
        except Exception:
            h_phase = float("nan")

        rows.append(dict(
            phase=name,
            T=T_out, Tsat=T_sat,
            U=U, rho=rho, mu=mu, Re=Re,
            alpha=alpha_single,
            cp=cp_phase, k=k_phase, MW=MW_val, h=h_phase,
        ))

    _print_table(
        "PROPRIETÀ DI USCITA - Fasi (T, Tsat, U, rho, mu, Re, alpha, Cp, k, MW, h)",
        cols, rows
    )

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
    - U_mix, rho_mix, mu_mix, Re_mix, Mach
    - sigma (tensione superficiale) valutata a T_sat(P2), Q=0 se bifase.
    """
    p2 = p2_bar * 1e5
    try:
        T_sat = cp.PropsSI("T", "P", p2, "Q", 1, fluid)
    except Exception:
        T_sat = out["T_out"]

    phase_out = out["phase_out"]

    # Tensione superficiale:
    # - ha senso solo se l'uscita è bifase (interfaccia liquido–vapore)
    # - usiamo I(T_sat, Q=0) [N/m]; in monofase restituiamo 0.0
    if phase_out == "two-phase":
        try:
            sigma = cp.PropsSI("I", "T", T_sat, "Q", 0, fluid)
        except Exception:
            sigma = float("nan")
    else:
        sigma = 0.0

    cols = [
        ("mdot [kg/s]",  "mdot",  12, ".6f"),
        ("fase",         "phase", 10, "s"),
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
        "RISULTATI PRINCIPALI - Portata phase-aware e proprietà miscela",
        cols, rows
    )

# ============================================================
#  MAIN
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
    parser.add_argument("--Cd-mode", type=str, default="user",
                        choices=["user", "geom"],
                        help="Modo di determinazione del Cd: "
                             "'user' = usa --Cd, 'geom' = stima da geometria.")
    parser.add_argument("--rD", type=float, default=0.05,
                        help="Edge radius ratio r/D per la stima geometrica del Cd "
                             "(usato solo se --Cd-mode=geom, ma riportato sempre in tabella).")
    parser.add_argument("--fluid", type=str, default="NitrousOxide",
                        help="Fluid name for CoolProp (default: NitrousOxide)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable SPI compressibility correction")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Isentropic exponent n for SPI (optional, e.g. 1.2)")

    args = parser.parse_args()

    fluid        = args.fluid
    p1_bar       = float(args.p1)
    p2_bar       = float(args.p2)
    T_line       = float(args.Tline)
    D            = float(args.D)
    L            = float(args.L)
    use_spi_comp = not args.no_compress
    spi_n        = args.spi_n
    Cd_mode      = args.Cd_mode
    r_over_D     = args.rD

    # ----- Determinazione Cd -----
    if Cd_mode == "geom":
        # Stima Cd (e Re di riferimento) dal core, usando la stessa logica del design tool
        Cd_geom, Re_ref = estimate_Cd_from_geometry(
            fluid=fluid,
            p1_bar=p1_bar,
            T_line=T_line,
            D=D,
            L=L,
            r_over_D=r_over_D,
        )
        Cd = Cd_geom
        Cd_source = f"estimated from geometry (r/D={r_over_D:.3f}, Re≈{Re_ref:.2e})"
    else:
        Cd = float(args.Cd)
        Cd_source = "user-provided Cd"
        Re_ref = None  # opzionale, non usato ma tenuto per completezza

    # ---- 1) Tabella input ----
    inputs = dict(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        D=D,
        L=L,
        Cd=Cd,
        Cd_mode=Cd_mode,
        Cd_source=Cd_source,
        r_over_D=r_over_D,
    )
    print_inputs_table(inputs)

    # ---- 2) Portate SPI / HEM / NHNE + phase-aware ----
    mdot_spi = _mdot_spi(fluid, p1_bar, p2_bar, T_line, D, Cd,
                         use_compress=use_spi_comp, n_isentropic=spi_n)
    mdot_hem = _mdot_hem(fluid, p1_bar, p2_bar, T_line, D, Cd)
    mdot_nhne, _ = _mdot_nhne(fluid, p1_bar, p2_bar, T_line, D, Cd,
                              L_over_D=(L / D if D > 0.0 else None),
                              K_RESIDENCE=0.0,
                              use_spi_compress=use_spi_comp,
                              spi_n=spi_n)

    mdot_pa, model_used = compute_mdot_phaseaware(
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

    print_mdot_table(mdot_spi, mdot_hem, mdot_nhne, mdot_pa, model_used)

    # ---- 3) Stato di uscita coerente con mdot phase-aware ----
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5

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
