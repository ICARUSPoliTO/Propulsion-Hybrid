"""
pyinjection_design.py
---------------------
Design tool for plain-orifice N2O injectors.

Questo script esplora una griglia 3D in:
  - r/D  (edge radius ratio, raccordo d'ingresso)
  - D    (diametro orifizio)
  - L/D  (length-to-diameter ratio)

Per ciascun punto:
  - stima un Cd = f(r/D, L/D, Re) coerente con la ricerca svolta
    (correlazione geometrica semplificata basata su letteratura e dataset
     interni per orifizi corti N2O/CO2)
  - calcola la portata per foro usando il backend 0D phase-aware (SPI / NHNE)
    implementato in pyinjection_core / V5
  - ricostruisce lo stato a valle tramite HEM + bilancio energetico
  - valuta la portata totale per Nh fori e il relativo errore rispetto al target.
"""
# Il modello di Cd(r/D, L/D, Re) è implementato in pyinjection_core.estimate_Cd_geom
# e usato tramite estimate_Cd_from_geometry(...).

from __future__ import annotations

# ================== ANTI-OVERSUBSCRIPTION ==================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ================== IMPORT BASE ==================
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ================== BACKEND 0D (V5 / CORE) ==================
try:
    # Backend termodinamico + portate dalla versione V5
    from pyinjection_core import (
    compute_mdot_phaseaware,
    nhne_out_state_from_mdot,
    estimate_Cd_from_geometry,  # nuovo helper high-level dal core
    )

    HAS_PHASEAWARE = True

    def solve_mdot_phaseaware(
        fluid: str,
        p1: float,
        p2: float,
        T_line: float,
        D: float,
        Cd: float,
        *,
        use_spi_compress: bool = True,
        spi_n: float | None = None,
        L_over_D: float | None = None,
    ):
        """
        Wrapper verso il backend V5: calcola mdot phase-aware e stato in uscita.

        Parametri
        ---------
        fluid : str
            Nome CoolProp del fluido (es. 'NitrousOxide').
        p1, p2 : float
            Pressioni a monte e valle [Pa].
        T_line : float
            Temperatura linea a monte [K].
        D : float
            Diametro dell'orifizio [m].
        Cd : float
            Coefficiente di scarico [-].
        use_spi_compress : bool
            Usa SPI compressibile.
        spi_n : float | None
            Esponente politropico per SPI compressibile; se None usa il default del backend.
        L_over_D : float | None
            Rapporto L/D del foro. Se fornito, viene convertito in L = D * L_over_D.

        Ritorna
        -------
        mdot_per_hole : float
            Portata per singolo foro [kg/s].
        model_used : str
            'SPI' oppure 'NHNE' (modello scelto dal phase-aware).
        info : dict
            Dizionario con info di uscita (fase, densità, Re, Mach, ecc.).
        """
        # Conversione pressioni Pa → bar per il backend V5
        p1_bar = p1 / 1e5
        p2_bar = p2 / 1e5

        # Lunghezza assoluta se disponibile
        L = D * L_over_D if L_over_D is not None else None

        # 1) Portata phase-aware (SPI o NHNE) dal core V5
        mdot_phase, model_used = compute_mdot_phaseaware(
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

        # 2) Stato di uscita coerente (equilibrio HEM + bilancio energia)
        out = nhne_out_state_from_mdot(
            fluid=fluid,
            p1=p1,
            p2=p2,
            T_line=T_line,
            D=D,
            mdot_nhne=mdot_phase,
            h1_hint=None,
        )

        # 3) Info per il design
        info = dict(
            phase_from_spi=out.get("phase_out", "unknown"),
            rho_out_spi=out.get("rho_mix", None),
            T_out=out.get("T_out", None),
            x_out=out.get("x_out", None),
            alpha_out=out.get("alpha_out", None),
            rho_l=out.get("rho_l", None),
            rho_v=out.get("rho_v", None),
            mu_mix=out.get("mu_mix", None),
            Re_out=out.get("Re_out", None),
            Mach=out.get("Mach", None),
        )

        return mdot_phase, model_used, info

except ImportError as e:
    HAS_PHASEAWARE = False
    import traceback
    traceback.print_exc()
    raise ImportError(
        "Impossibile importare 'compute_mdot_phaseaware' / 'nhne_out_state_from_mdot' "
        "da pyinjection_core.py.\n"
        "Assicurati che pyinjection_core.py sia nello stesso folder o nel PYTHONPATH."
    ) from e


# ================== EVALUAZIONE DI UN CANDIDATO ==================
@dataclass
class CandidateResult:
    D: float
    L: float
    r_over_D: float
    L_over_D: float
    Re: float
    Cd: float
    mdot_total: float
    mdot_per_hole: float
    nh: int
    rho_l: float
    mu_l: float
    note: str = ""


def evaluate_candidate(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    nholes: int,
    r_over_D: float,
    D: float,
    L: float,
    Cd_fixed: Optional[float] = None,
) -> CandidateResult:
    """
    Valuta un candidato (r/D, D, L) per un certo numero di fori.

    Usa il backend phase-aware (SPI / NHNE) di V5 per calcolare la portata
    per foro e ricostruire le proprietà a valle (HEM).

    Se Cd_fixed è fornito, bypassa la stima geometrica e usa quel valore
    (tipicamente da CFD o esperimento). Altrimenti il Cd è stimato dal core
    in base a geometria e Re caratteristico.
    """
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5
    L_over_D = L / D

    # --- Cd + proprietà di riferimento dal core ---
    # (il core calcola rho_l, mu_l, Re_char e decide se usare Cd_input o Cd_geom)
    Cd_used, Re_char, rho_l, mu_l = estimate_Cd_from_geometry(
        fluid=fluid,
        p1=p1,
        T_line=T_line,
        D=D,
        L=L,
        r_over_D=r_over_D,
        Cd_input=Cd_fixed,   # None => stima geometrica, >0 => usa dato utente
    )

    # Area foro (serve dopo per Re_eff a posteriori)
    A_hole = 0.25 * math.pi * D**2

    # --- Portata phase-aware per foro ---
    mdot_per_hole, model_used, info = solve_mdot_phaseaware(
        fluid=fluid,
        p1=p1,
        p2=p2,
        T_line=T_line,
        D=D,
        Cd=Cd_used,
        use_spi_compress=True,
        spi_n=1.2,
        L_over_D=L_over_D,
    )

    phase_out = info.get("phase_from_spi", "unknown")
    rho_out = info.get("rho_out_spi", rho_l)  # densità miscela per Re_eff

    # --- Re effettivo basato sulla portata finale ---
    if mdot_per_hole > 0.0 and rho_out > 0.0:
        U_eff = mdot_per_hole / (rho_out * A_hole)
        Re_eff = rho_out * abs(U_eff) * D / max(mu_l, 1e-9)
    else:
        Re_eff = float("nan")

    return CandidateResult(
        D=D,
        L=L,
        r_over_D=r_over_D,
        L_over_D=L_over_D,
        Re=Re_eff,
        Cd=Cd_used,
        mdot_total=mdot_per_hole * nholes,
        mdot_per_hole=mdot_per_hole,
        nh=nholes,
        rho_l=rho_l,
        mu_l=mu_l,
        note=f"model={model_used}, phase_out={phase_out}, Re_char={Re_char:.3e}",
    )

def _eval_candidate_wrapper(args_tuple):
    """Wrapper per parallelizzazione (ThreadPoolExecutor.map)."""
    return evaluate_candidate(*args_tuple)


# ================== LOOP DI DESIGN SU GRIGLIA 3D ==================
def design_from_mdot(
    fluid: str,
    p1_bar: float,
    p2_bar: float,
    T_line: float,
    mdot_target: float,
    nholes: int,
    rD_min: float,
    rD_max: float,
    n_rD: int,
    D_min: float,
    D_max: float,
    n_D: int,
    L_over_D_min: float,
    L_over_D_max: float,
    n_LD: int,
    n_workers: int = 1,
    Cd_fixed: Optional[float] = None,
) -> List[CandidateResult]:
    """
    Scansiona una griglia 3D in (r/D, D, L/D) e calcola mdot_total per ciascun candidato.
    Parallelizza via ThreadPoolExecutor se n_workers > 1.

    Se Cd_fixed è fornito, usa quel valore di Cd per tutti i candidati;
    altrimenti usa la correlazione estimate_Cd(r/D,L/D,Re).
    """
    if not HAS_PHASEAWARE:
        raise RuntimeError(
            "Backend phase-aware non disponibile: controlla pyinjection_core.py."
        )

    if D_min <= 0.0 or D_max <= 0.0 or D_max <= D_min:
        raise ValueError("Range D_min, D_max non valido.")
    if n_D < 2 or n_LD < 2 or n_rD < 1:
        raise ValueError("Servono almeno: n_D>=2, n_LD>=2, n_rD>=1.")

    D_values  = np.linspace(D_min, D_max, n_D)
    LD_values = np.linspace(L_over_D_min, L_over_D_max, n_LD)
    rD_values = np.linspace(rD_min, rD_max, n_rD) if n_rD > 1 else np.array([rD_min])

    # Costruisco la lista di parametri per ogni punto della griglia
    param_list = []
    for r_over_D in rD_values:
        for D in D_values:
            for LD in LD_values:
                L = LD * D
                param_list.append((
                    fluid,
                    p1_bar,
                    p2_bar,
                    T_line,
                    nholes,
                    float(r_over_D),
                    float(D),
                    float(L),
                    Cd_fixed,
                ))

    results: List[CandidateResult] = []

    if n_workers is None or n_workers <= 1:
        # Esecuzione seriale
        for params in param_list:
            try:
                res = _eval_candidate_wrapper(params)
            except Exception as e:
                res = CandidateResult(
                    D=params[6],
                    L=params[7],
                    r_over_D=params[5],
                    L_over_D=params[7]/params[6] if params[6] > 0 else float("nan"),
                    Re=float("nan"),
                    Cd=float("nan"),
                    mdot_total=0.0,
                    mdot_per_hole=0.0,
                    nh=nholes,
                    rho_l=float("nan"),
                    mu_l=float("nan"),
                    note=f"ERROR: {e}",
                )
            results.append(res)
    else:
        # Parallelizzazione con thread
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_eval_candidate_wrapper, p) for p in param_list]
            for p, fut in zip(param_list, futures):
                try:
                    res = fut.result()
                except Exception as e:
                    res = CandidateResult(
                        D=p[6],
                        L=p[7],
                        r_over_D=p[5],
                        L_over_D=p[7]/p[6] if p[6] > 0 else float("nan"),
                        Re=float("nan"),
                        Cd=float("nan"),
                        mdot_total=0.0,
                        mdot_per_hole=0.0,
                        nh=nholes,
                        rho_l=float("nan"),
                        mu_l=float("nan"),
                        note=f"ERROR: {e}",
                    )
                results.append(res)

    return results


def _rel_err(res: CandidateResult, mdot_target: float) -> float:
    """Errore relativo (0–1) su mdot_total, oppure inf se non definito."""
    if mdot_target <= 0.0 or res.mdot_total <= 0.0:
        return float("inf")
    return abs(res.mdot_total - mdot_target) / mdot_target


def select_best_candidates(results: List[CandidateResult],
                           mdot_target: float,
                           topk: int = 10) -> List[CandidateResult]:
    """
    Ordina i candidati per errore relativo sulla mdot totale
    e restituisce i migliori topk.
    """
    if mdot_target <= 0.0:
        raise ValueError("mdot_target deve essere > 0.")

    results_sorted = sorted(results, key=lambda r: _rel_err(r, mdot_target))
    return results_sorted[:topk]


def get_feasible_candidates(results: List[CandidateResult],
                            mdot_target: float,
                            tol_rel_perc: float) -> List[CandidateResult]:
    """
    Restituisce tutti i candidati che soddisfano:
        |mdot - mdot_target| / mdot_target <= tol_rel_perc/100.
    """
    if mdot_target <= 0.0:
        return []

    tol = tol_rel_perc / 100.0
    feas = [r for r in results if _rel_err(r, mdot_target) <= tol]
    return sorted(feas, key=lambda r: _rel_err(r, mdot_target))


# ================== PLOT 3D RISULTATI ==================
def plot_design_map_3d(results: List[CandidateResult],
                       mdot_target: float,
                       tol_rel_perc: float) -> None:
    """
    Crea un grafico 3D scatter nel volume (D [mm], L/D, r/D),
    colorato in base al rapporto mdot_total / mdot_target.
    Evidenzia anche i punti che soddisfano il vincolo sulla portata.
    """
    if mdot_target <= 0.0:
        print("plot_design_map_3d: mdot_target <= 0, salto il plot.")
        return

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib non disponibile, nessun grafico generato.")
        return

    Dmm     = np.array([r.D * 1e3 for r in results])
    LD_arr  = np.array([r.L_over_D for r in results])
    rD_arr  = np.array([r.r_over_D for r in results])

    ratio = np.array([
        r.mdot_total / mdot_target if (mdot_target > 0.0 and r.mdot_total > 0.0)
        else np.nan
        for r in results
    ])

    mask_all = np.isfinite(ratio)
    if not np.any(mask_all):
        print("plot_design_map_3d: nessun valore valido per la portata, salto il plot.")
        return

    # Maschera per i punti che rispettano la tolleranza
    tol = tol_rel_perc / 100.0
    err_rel = np.abs(ratio - 1.0)
    mask_feas = mask_all & (err_rel <= tol)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter di tutti i punti
    sc = ax.scatter(Dmm[mask_all], LD_arr[mask_all], rD_arr[mask_all],
                    c=ratio[mask_all], marker="o", alpha=0.4,
                    vmin=0.5, vmax=1.5)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("mdot_total / mdot_target")

    # Punti che soddisfano il requisito sulla portata (evidenziati)
    if np.any(mask_feas):
        ax.scatter(Dmm[mask_feas], LD_arr[mask_feas], rD_arr[mask_feas],
                   c="red", marker="^", s=40, label=f"|err_rel| <= {tol_rel_perc:.3f}%")
        ax.legend(loc="best")

    ax.set_xlabel("D [mm]")
    ax.set_ylabel("L/D [-]")
    ax.set_zlabel("r/D [-]")
    ax.set_title("Design map 3D: mass flow ratio")

    plt.tight_layout()
    plt.show()


# ================== STAMPA RISULTATI ==================
def print_candidate_table(cands: List[CandidateResult], mdot_target: float) -> None:
    """Stampa una tabella compatta dei candidati selezionati."""
    if not cands:
        print("(nessun candidato)")
        return

    print(
        "   # |   D [mm] |   L [mm] |  L/D |  r/D |    Re [-]    |   Cd  | nh | mdot_tot [kg/s] | err_rel [%] | note"
    )
    print("-" * 118)

    for i, c in enumerate(cands, start=1):
        if mdot_target > 0.0:
            err = abs(c.mdot_total - mdot_target) / mdot_target * 100.0
        else:
            err = float("nan")
        print(
            f"{i:4d} |"
            f"{c.D*1e3:9.3f} |"
            f"{c.L*1e3:9.3f} |"
            f"{c.L_over_D:5.2f} |"
            f"{c.r_over_D:5.3f} |"
            f"{c.Re:12.3e} |"
            f"{c.Cd:6.3f} |"
            f"{c.nh:3d} |"
            f"{c.mdot_total:15.5f} |"
            f"{err:10.3f} | "
            f"{c.note}"
        )

    print("-" * 118)


def print_best_candidate(c: CandidateResult, mdot_target: float) -> None:
    """Stampa una piccola tabella riassuntiva per il miglior candidato globale."""
    if mdot_target > 0.0:
        err = abs(c.mdot_total - mdot_target) / mdot_target * 100.0
    else:
        err = float("nan")

    print("\n=== Best candidate (global minimum error) ===")
    print(f"D [mm]     : {c.D*1e3:.3f}")
    print(f"L [mm]     : {c.L*1e3:.3f}")
    print(f"L/D        : {c.L_over_D:.3f}")
    print(f"r/D        : {c.r_over_D:.4f}")
    print(f"Cd [-]     : {c.Cd:.4f}")
    print(f"Nh [-]     : {c.nh:d}")
    print(f"mdot_tot   : {c.mdot_total:.6f} kg/s")
    print(f"err_rel    : {err:.3f} %")
    print(f"Re         : {c.Re:.3e}")
    print(f"rho_l@P1   : {c.rho_l:.3f} kg/m^3")
    print(f"mu_l@P1    : {c.mu_l:.3e} Pa·s")
    print(f"note       : {c.note}")
    print("============================================\n")


# ================== MAIN / CLI ==================
def main():
    parser = argparse.ArgumentParser(
        description="Plain-orifice N2O injector design tool (phase-aware V5 backend)."
    )

    parser.add_argument("--fluid", type=str, default="NitrousOxide",
                        help="Working fluid (CoolProp name). Default: NitrousOxide")

    parser.add_argument("--p1-bar", type=float, default=55.0,
                        help="Inlet pressure P1 [bar]. Default: 55")
    parser.add_argument("--p2-bar", type=float, default=43.0,
                        help="Chamber/back pressure P2 [bar]. Default: 43")
    parser.add_argument("--T-line", type=float, default=288.0,
                        help="Feed line temperature [K]. Default: 288")

    parser.add_argument("--mdot-target", type=float, default=0.140,
                        help="Target TOTAL mass flow [kg/s]. Default: 0.140")

    parser.add_argument("--Nh", type=int, default=1,
                        help="Number of injector holes. Default: 1")

    # Range r/D
    parser.add_argument("--rD-min", type=float, default=0.05,
                        help="Minimum r/D. Default: 0.05")
    parser.add_argument("--rD-max", type=float, default=0.35,
                        help="Maximum r/D. Default: 0.35")
    parser.add_argument("--n-rD",   type=int,   default=7,
                        help="Number of r/D samples. Default: 7")

    # Range D
    parser.add_argument("--Dmin-mm", type=float, default=0.5,
                        help="Minimum hole diameter [mm]. Default: 0.5")
    parser.add_argument("--Dmax-mm", type=float, default=3.5,
                        help="Maximum hole diameter [mm]. Default: 3.5")
    parser.add_argument("--nD", type=int, default=25,
                        help="Number of D samples in [Dmin,Dmax]. Default: 25")

    # Range L/D
    parser.add_argument("--LD-min", type=float, default=2.0,
                        help="Minimum L/D ratio. Default: 2.0")
    parser.add_argument("--LD-max", type=float, default=12.0,
                        help="Maximum L/D ratio. Default: 12.0")
    parser.add_argument("--nLD", type=int, default=10,
                        help="Number of L/D samples. Default: 10")

    parser.add_argument("--tol-rel-perc", type=float, default=3.0,
                        help="Relative mass-flow error tolerance [%%] for 'good' points. Default: 3.0")

    parser.add_argument("--topk", type=int, default=10,
                        help="Number of candidates to print in the table. Default: 10")

    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional CSV file path for exporting full grid results.")

    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of the 3D design map.")

    parser.add_argument("--n-workers", type=int, default=24,
                        help="Number of worker threads for parallel evaluation (1 = serial).")

    parser.add_argument("--Cd-fixed", type=float, default=None,
                        help="If provided, use this fixed Cd (per hole) instead of geometric estimation.")

    args = parser.parse_args()

    fluid       = args.fluid
    p1_bar      = args.p1_bar
    p2_bar      = args.p2_bar
    T_line      = args.T_line
    mdot_target = args.mdot_target
    Nh          = args.Nh

    rD_min = args.rD_min
    rD_max = args.rD_max
    n_rD   = args.n_rD

    Dmin = args.Dmin_mm * 1e-3
    Dmax = args.Dmax_mm * 1e-3
    nD   = args.nD
    LD_min = args.LD_min
    LD_max = args.LD_max
    nLD    = args.nLD

    tol_rel_perc = args.tol_rel_perc
    n_workers    = args.n_workers
    Cd_fixed     = args.Cd_fixed

    print("=== Injector design setup ===")
    print(f"Fluid        : {fluid}")
    print(f"P1           : {p1_bar:.2f} bar")
    print(f"P2           : {p2_bar:.2f} bar")
    print(f"T_line       : {T_line:.2f} K")
    print(f"mdot target  : {mdot_target:.5f} kg/s (total)")
    print(f"Nh (holes)   : {Nh}")
    print(f"r/D range    : {rD_min:.3f}–{rD_max:.3f}  (n_rD = {n_rD})")
    print(f"D range      : {args.Dmin_mm:.3f}–{args.Dmax_mm:.3f} mm  (nD = {nD})")
    print(f"L/D range    : {LD_min:.2f}–{LD_max:.2f}  (nLD = {nLD})")
    print(f"tol_rel      : {tol_rel_perc:.3f} %")
    if Cd_fixed is not None:
        print(f"Cd (fixed)   : {Cd_fixed:.4f}")
    else:
        print("Cd mode      : estimated from geometry (r/D, L/D, Re)")
    print(f"Workers      : {n_workers}")
    print("")

    results = design_from_mdot(
        fluid=fluid,
        p1_bar=p1_bar,
        p2_bar=p2_bar,
        T_line=T_line,
        mdot_target=mdot_target,
        nholes=Nh,
        rD_min=rD_min,
        rD_max=rD_max,
        n_rD=n_rD,
        D_min=Dmin,
        D_max=Dmax,
        n_D=nD,
        L_over_D_min=LD_min,
        L_over_D_max=LD_max,
        n_LD=nLD,
        n_workers=n_workers,
        Cd_fixed=Cd_fixed,
    )

    if not results:
        print("Nessun risultato generato (lista vuota).")
        return

    # Migliori candidati globali
    best_overall = select_best_candidates(results, mdot_target, topk=args.topk)

    # Punti che soddisfano il requisito sulla portata
    feasible = get_feasible_candidates(results, mdot_target, tol_rel_perc)

    print("=== Selected design candidates ===")
    if feasible:
        # stampo i migliori topk fra quelli entro la tolleranza
        to_print = feasible[:args.topk]
        print_candidate_table(to_print, mdot_target)
        if len(feasible) > args.topk:
            print(f"... e altri {len(feasible) - args.topk} punti soddisfano il requisito.\n")
    else:
        # nessun punto entro la tolleranza: stampo i migliori globali
        print("(nessun candidato soddisfa la tolleranza su mdot; stampo i migliori globali)\n")
        print_candidate_table(best_overall, mdot_target)

    # Tabellina riassuntiva per il miglior candidato globale
    best_cand = best_overall[0]
    print_best_candidate(best_cand, mdot_target)

    # Export CSV opzionale
    if args.csv_out is not None:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "D[m]", "L[m]", "r_over_D", "L_over_D", "Re", "Cd",
                "nh", "mdot_total[kg/s]", "mdot_per_hole[kg/s]",
                "rho_l[kg/m3]", "mu_l[Pa*s]", "note"
            ])
            for c in results:
                w.writerow([
                    c.D, c.L, c.r_over_D, c.L_over_D, c.Re, c.Cd,
                    c.nh, c.mdot_total, c.mdot_per_hole,
                    c.rho_l, c.mu_l, c.note
                ])
        print(f"\nFull grid results written to: {args.csv_out}\n")

    # Plot 3D
    if not args.no_plot:
        plot_design_map_3d(results, mdot_target, tol_rel_perc)


if __name__ == "__main__":
    main()
