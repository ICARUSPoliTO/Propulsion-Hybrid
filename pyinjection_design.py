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
import mplcursors
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
    mdot_target: float,
    nholes: int,
    r_over_D: float,
    D: float,
    L: float,
    D_pipe: float,
    Cd_fixed: Optional[float] = None,
) -> CandidateResult:
    """
    Valuta un candidato (r/D, D, L) per un certo numero di fori.

    Usa il backend phase-aware (SPI / NHNE) di V5 per calcolare la portata
    per foro e ricostruire le proprietà a valle (HEM).

    Parametri
    ---------
    fluid : str
        Nome CoolProp del fluido.
    p1_bar, p2_bar : float
        Pressioni a monte e valle [bar].
    T_line : float
        Temperatura della linea a monte [K].
    mdot_target : float
        Portata TOTALE target [kg/s] (usata per il Re nella stima di Cd).
    nholes : int
        Numero di fori.
    r_over_D : float
        Rapporto r/D (raccordo ingresso).
    D : float
        Diametro orifizio [m].
    L : float
        Lunghezza orifizio [m].
    D_pipe : float
        Diametro del condotto a monte (manifold) [m].
    Cd_fixed : Optional[float]
        Se fornito > 0, override di Cd (es. da CFD/esperimenti).

    Ritorna
    -------
    CandidateResult
        Risultato con mdot per foro, mdot totale, Cd stimato e Re caratteristico.
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
        D_orif=D,
        L=L,
        r_over_D=r_over_D,
        D_pipe=D_pipe,
        mdot_target=mdot_target,
        Cd_input=Cd_fixed,   # None => stima geometrica, >0 => usa dato utente
    )

    # --- Portata phase-aware + stato di uscita ---
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

    # Densità e viscosità miscela per Re_eff
    rho_mix = info.get("rho_out_spi", rho_l)
    mu_mix  = info.get("mu_mix", mu_l)

    A_hole = math.pi * (D**2) / 4.0

    # Re effettivo: se il backend un giorno ti restituisce U_out lo usi,
    # altrimenti lo ricavi dalla portata.
    U_mix = info.get("U_out", None)

    if (U_mix is not None) and (rho_mix is not None) and (rho_mix > 0.0):
        Re_eff = rho_mix * abs(U_mix) * D / max(mu_mix, 1e-9)
    elif mdot_per_hole > 0.0 and (rho_mix is not None) and (rho_mix > 0.0):
        U_eff = mdot_per_hole / (rho_mix * A_hole)
        Re_eff = rho_mix * abs(U_eff) * D / max(mu_mix, 1e-9)
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
    D_pipe: float,
    n_workers: int = 1,
    Cd_fixed: Optional[float] = None,
) -> List[CandidateResult]:
    """
    Esegue una ricerca su griglia 3D in (r/D, D, L/D) per trovare
    le geometrie che soddisfano la portata target.

    Parametri
    ---------
    fluid, p1_bar, p2_bar, T_line : come da main().
    mdot_target : float
        Portata TOTALE target [kg/s].
    nholes : int
        Numero di fori.
    rD_min, rD_max, n_rD : range e numero di campioni per r/D.
    D_min, D_max, n_D : range e numero di campioni per D [m].
    L_over_D_min, L_over_D_max, n_LD : range e numero di campioni per L/D.
    D_pipe : float
        Diametro del condotto a monte [m] (usato per β = D_orif / D_pipe).
    n_workers : int
        Numero di thread per la valutazione in parallelo.
    Cd_fixed : Optional[float]
        Se non None, Cd viene fissato a questo valore (bypass della correlazione).

    Ritorna
    -------
    results : List[CandidateResult]
        Lista con tutti i candidati valutati.
    """
    # Discretizzazione della griglia
    D_values  = np.linspace(D_min, D_max, n_D)
    LD_values = np.linspace(L_over_D_min, L_over_D_max, n_LD)
    rD_values = np.linspace(rD_min, rD_max, n_rD) if n_rD > 1 else np.array([rD_min])

    # Lista di parametri per ogni punto della griglia
    param_list = []
    for r_over_D in rD_values:
        for D in D_values:
            for LD in LD_values:
                L = LD * D
                param_list.append((
                    fluid,           # 0
                    p1_bar,          # 1
                    p2_bar,          # 2
                    T_line,          # 3
                    mdot_target,     # 4
                    nholes,          # 5
                    float(r_over_D), # 6
                    float(D),        # 7
                    float(L),        # 8
                    float(D_pipe),   # 9
                    Cd_fixed,        # 10
                ))

    results: List[CandidateResult] = []

    if n_workers is None or n_workers <= 1:
        # Esecuzione seriale
        for params in param_list:
            try:
                res = _eval_candidate_wrapper(params)
            except Exception as e:
                # In caso di errore, inserisco un CandidateResult "vuoto" con nota
                res = CandidateResult(
                    D=params[7],
                    L=params[8],
                    r_over_D=params[6],
                    L_over_D=params[8] / params[7] if params[7] > 0 else float("nan"),
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
        import multiprocessing
        n_workers = multiprocessing.cpu_count() - 1
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_eval_candidate_wrapper, p) for p in param_list]
            for p, fut in zip(param_list, futures):
                try:
                    res = fut.result()
                except Exception as e:
                    res = CandidateResult(
                        D=p[7],
                        L=p[8],
                        r_over_D=p[6],
                        L_over_D=p[8] / p[7] if p[7] > 0 else float("nan"),
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
def plot_cd_vs_ratio_by_diameter(results: List[CandidateResult],
                                 mdot_target: float,
                                 tol_rel_perc: float) -> None:
    """
    Grafico 'in stile fronte di Pareto' per i soli candidati che soddisfano
    il requisito di portata entro una certa tolleranza.

    Ogni punto mostra un tooltip interattivo con:
        - L/D
        - r/D
        - Re
        - Cd
        - mdot_ratio
        - D
    """
    if mdot_target <= 0.0:
        print("plot_cd_vs_ratio_by_diameter: mdot_target <= 0, salto il plot.")
        return

    # Seleziona i candidati che rispettano il vincolo sulla portata
    feas = get_feasible_candidates(results, mdot_target, tol_rel_perc)
    if not feas:
        print("plot_cd_vs_ratio_by_diameter: nessun candidato entro la tolleranza, nessun grafico.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponibile, nessun grafico generato.")
        return

    # Diametri distinti (in m)
    unique_D = sorted({round(c.D, 9) for c in feas})

    fig, ax = plt.subplots()

    cmap = plt.cm.get_cmap("tab10")

    # Per salvare info dei punti
    scatter_plots = []

    for idx, D in enumerate(unique_D):
        group = [c for c in feas if abs(c.D - D) <= 1e-9]
        if not group:
            continue

        x_cd = [c.Cd for c in group]
        y_ratio = [c.mdot_total / mdot_target for c in group]

        color = cmap(idx % 10)
        label = f"D = {D*1e3:.3f} mm (n={len(group)})"

        sc = ax.scatter(x_cd, y_ratio, marker="o", alpha=0.8, label=label, color=color)

        # Salviamo i dati associati a ogni singolo punto
        sc._candidate_info = group
        scatter_plots.append(sc)

    # Linea target
    ax.axhline(1.0, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Cd [-]")
    ax.set_ylabel("mdot_total / mdot_target [-]")
    ax.set_title(f"Configurazioni entro {tol_rel_perc:.3f}% sulla portata target")

    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize="small")

    # -------------------------
    #  TOOLTIP INTERATTIVO
    # -------------------------
    cursor = mplcursors.cursor(scatter_plots, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        sc = sel.artist
        index = sel.index
        cand = sc._candidate_info[index]

        text = (
            f"D = {cand.D*1e3:.3f} mm\n"
            f"L/D = {cand.L_over_D:.3f}\n"
            f"r/D = {cand.r_over_D:.3f}\n"
            f"Re = {cand.Re:.2e}\n"
            f"Cd = {cand.Cd:.3f}\n"
            f"mdot_ratio = {cand.mdot_total/mdot_target:.4f}"
        )

        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

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

# ================== SEMPLICE INTERFACCIA GRAFICA (Tkinter) ==================
def run_gui():
    """
    Piccola GUI Tkinter per inserire gli input del design e lanciare la griglia.

    Apre una finestra con i campi principali; alla pressione di "Run design"
    esegue design_from_mdot(...) e mostra il grafico Cd vs mdot_ratio.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("PyInjection – Injector Design GUI")

    # --------- campi e valori di default (gli stessi del CLI) ----------
    fields = [
        ("Fluid (CoolProp)",       "fluid",        "NitrousOxide"),
        ("P1 [bar]",               "p1_bar",       "55.0"),
        ("P2 [bar]",               "p2_bar",       "43.0"),
        ("T_line [K]",             "T_line",       "288.0"),
        ("mdot target TOTAL [kg/s]","mdot_target", "0.140"),
        ("Nh (number of holes)",   "Nh",           "1"),
        ("r/D min",                "rD_min",       "0.05"),
        ("r/D max",                "rD_max",       "0.35"),
        ("n_rD",                   "n_rD",         "7"),
        ("D_min [mm]",             "Dmin_mm",      "0.5"),
        ("D_max [mm]",             "Dmax_mm",      "3.5"),
        ("nD",                     "nD",           "25"),
        ("D_pipe [mm]",            "Dpipe_mm",     "5.0"),
        ("L/D min",                "LD_min",       "2.0"),
        ("L/D max",                "LD_max",       "12.0"),
        ("n_LD",                   "nLD",          "10"),
        ("tol. rel. [%]",          "tol_rel_perc", "3.0"),
        ("n_workers",              "n_workers",    "4"),
        ("Cd fixed (blank = auto)","Cd_fixed",     ""),
    ]

    entries: dict[str, tk.Entry] = {}

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    for row, (label_text, key, default) in enumerate(fields):
        lab = ttk.Label(main_frame, text=label_text + ":")
        lab.grid(row=row, column=0, sticky="e", padx=5, pady=2)

        ent = ttk.Entry(main_frame, width=18)
        ent.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        ent.insert(0, default)
        entries[key] = ent

    # --------- callback del bottone "Run design" ----------
    def on_run():
        try:
            fluid        = entries["fluid"].get().strip() or "NitrousOxide"
            p1_bar       = float(entries["p1_bar"].get() or 55.0)
            p2_bar       = float(entries["p2_bar"].get() or 43.0)
            T_line       = float(entries["T_line"].get() or 288.0)
            mdot_target  = float(entries["mdot_target"].get() or 0.140)
            Nh           = int(entries["Nh"].get() or 1)

            rD_min       = float(entries["rD_min"].get() or 0.05)
            rD_max       = float(entries["rD_max"].get() or 0.35)
            n_rD         = int(entries["n_rD"].get() or 7)

            Dmin_mm      = float(entries["Dmin_mm"].get() or 0.5)
            Dmax_mm      = float(entries["Dmax_mm"].get() or 3.5)
            nD           = int(entries["nD"].get() or 25)

            Dpipe_mm     = float(entries["Dpipe_mm"].get() or 5.0)

            LD_min       = float(entries["LD_min"].get() or 2.0)
            LD_max       = float(entries["LD_max"].get() or 12.0)
            nLD          = int(entries["nLD"].get() or 10)

            tol_rel_perc = float(entries["tol_rel_perc"].get() or 3.0)
            n_workers    = int(entries["n_workers"].get() or 4)

            Cd_fixed_str = entries["Cd_fixed"].get().strip()
            Cd_fixed     = float(Cd_fixed_str) if Cd_fixed_str else None

        except ValueError as e:
            messagebox.showerror("Input error", f"Valore non valido: {e}")
            return

        # Conversioni di unità
        D_min   = Dmin_mm  * 1e-3
        D_max   = Dmax_mm  * 1e-3
        D_pipe  = Dpipe_mm * 1e-3

        # Esegui il design vero e proprio
        try:
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
                D_min=D_min,
                D_max=D_max,
                n_D=nD,
                L_over_D_min=LD_min,
                L_over_D_max=LD_max,
                n_LD=nLD,
                D_pipe=D_pipe,
                n_workers=n_workers,
                Cd_fixed=Cd_fixed,
            )
        except Exception as e:
            messagebox.showerror("Design error", f"Errore durante il design:\n{e}")
            return

        if not results:
            messagebox.showwarning("Design", "Nessun risultato generato (lista vuota).")
            return

        best_overall = select_best_candidates(results, mdot_target, topk=10)
        feasible     = get_feasible_candidates(results, mdot_target, tol_rel_perc)

        print("\n=== RISULTATI (GUI) – migliori candidati ===")
        if feasible:
            to_print = feasible[:10]
            print_candidate_table(to_print, mdot_target)
            print(f"... e altri {max(0, len(feasible) - 10)} punti soddisfano il requisito.\n")
        else:
            print("(GUI) Nessun candidato entro la tolleranza; stampo i migliori globali\n")
            print_candidate_table(best_overall, mdot_target)

        # Messaggio breve nella GUI
        if feasible:
            messagebox.showinfo(
                "Design completed",
                f"Design completato.\nCandidati entro tol.: {len(feasible)}"
            )
        else:
            messagebox.showinfo(
                "Design completed",
                "Design completato.\nNessun candidato entro la tolleranza."
            )

        # Plot
        try:
            plot_cd_vs_ratio_by_diameter(results, mdot_target, tol_rel_perc)
        except Exception as e:
            messagebox.showwarning("Plot error", f"Errore nella generazione del grafico:\n{e}")

    run_button = ttk.Button(main_frame, text="Run design", command=on_run)
    run_button.grid(row=len(fields), column=0, columnspan=2, pady=10)

    root.mainloop()

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

        # Range D (diametro del foro)
    parser.add_argument("--Dmin-mm", type=float, default=0.5,
                        help="Minimum hole diameter [mm]. Default: 0.5")
    parser.add_argument("--Dmax-mm", type=float, default=3.5,
                        help="Maximum hole diameter [mm]. Default: 3.5")
    parser.add_argument("--nD", type=int, default=25,
                        help="Number of D samples in [Dmin,Dmax]. Default: 25")

    # Diametro del condotto a monte (manifold) per β = D_orif / D_pipe
    parser.add_argument("--Dpipe-mm", type=float, default=5.0,
                        help="Manifold (upstream pipe) diameter [mm] for β and RHG. Default: 5.0")

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

    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of worker threads for parallel evaluation. Default: 1")

    parser.add_argument("--Cd-fixed", type=float, default=None,
                        help="If provided, use this fixed Cd instead of geometry-based estimate.")
    
    parser.add_argument("--gui", action="store_true",
                        help="Lancia una semplice GUI Tkinter per l'inserimento degli input.")


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

    Dmin   = args.Dmin_mm * 1e-3
    Dmax   = args.Dmax_mm * 1e-3
    nD     = args.nD

    D_pipe = args.Dpipe_mm * 1e-3

    LD_min = args.LD_min
    LD_max = args.LD_max
    nLD    = args.nLD

    tol_rel_perc = args.tol_rel_perc
    n_workers    = args.n_workers
    Cd_fixed     = args.Cd_fixed

    # Se richiesto, lancia solo la GUI e termina
    if args.gui:
        run_gui()
        return


    print("=== Injector design setup ===")
    print(f"Fluid        : {fluid}")
    print(f"P1           : {p1_bar:.2f} bar")
    print(f"P2           : {p2_bar:.2f} bar")
    print(f"T_line       : {T_line:.2f} K")
    print(f"mdot target  : {mdot_target:.5f} kg/s (total)")
    print(f"Nh (holes)   : {Nh:d}")
    print(f"r/D range    : {rD_min:.3f}–{rD_max:.3f}  (n_rD = {n_rD})")
    print(f"D range      : {args.Dmin_mm:.3f}–{args.Dmax_mm:.3f} mm  (nD = {nD})")
    print(f"D_pipe       : {args.Dpipe_mm:.3f} mm (manifold)")
    print(f"L/D range    : {LD_min:.2f}–{LD_max:.2f}  (nLD = {nLD})")
    print(f"tol_rel      : {tol_rel_perc:.3f} %")
    if Cd_fixed is not None:
        print(f"Cd (fixed)   : {Cd_fixed:.4f}")
    else:
        print("Cd mode      : estimated from geometry (RHG + r/D + Darcy)")
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
        D_pipe=D_pipe,
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

    # Plot "in stile Pareto" Cd vs mdot_ratio per i diametri che soddisfano il vincolo
    if not args.no_plot:
        plot_cd_vs_ratio_by_diameter(results, mdot_target, tol_rel_perc)

if __name__ == "__main__":
    import sys

    # Se l'utente lancia il file SENZA argomenti => apri direttamente la GUI
    if len(sys.argv) == 1:
        run_gui()
    else:
        main()
