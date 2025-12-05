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
import multiprocessing

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

def _auto_n_workers(n_workers: int | None) -> int:
    """
    Return an effective number of worker processes based on the available CPU cores.

    - If n_workers is None or <= 0 → use (cpu_count - 1), at least 1.
    - Otherwise clamp n_workers to at most (cpu_count - 1).
    """
    try:
        max_hw = max(1, multiprocessing.cpu_count() - 1)
    except Exception:
        max_hw = 1

    if n_workers is None:
        return max_hw

    try:
        nw = int(n_workers)
    except Exception:
        return max_hw

    if nw <= 0:
        return max_hw

    return min(nw, max_hw)

    """
    Return an effective number of worker threads based on the available CPU cores.

    - If n_workers is None or <= 0 → use (cpu_count - 1), at least 1.
    - Otherwise clamp n_workers to at most (cpu_count - 1).
    """
    try:
        max_hw = max(1, multiprocessing.cpu_count() - 1)
    except Exception:
        max_hw = 1

    if n_workers is None:
        return max_hw

    try:
        nw = int(n_workers)
    except Exception:
        return max_hw

    if nw <= 0:
        return max_hw

    return min(nw, max_hw)

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
    use_spi_compress: bool = True,
    spi_n: float | None = None,
) -> CandidateResult:
    """
    Evaluate one candidate (r/D, D, L) for a given number of holes.

    It uses the phase-aware backend (SPI / NHNE) from V5 to compute:
      - per-hole mass flow
      - outlet properties (HEM-based)
      - effective Reynolds number.

    Parameters
    ----------
    fluid : str
        CoolProp fluid name.
    p1_bar, p2_bar : float
        Upstream and downstream pressures [bar].
    T_line : float
        Feeding line temperature [K].
    mdot_target : float
        TOTAL target mass flow [kg/s].
    nholes : int
        Number of identical holes.
    r_over_D : float
        Edge-radius ratio.
    D : float
        Orifice diameter [m].
    L : float
        Orifice length [m].
    D_pipe : float
        Upstream manifold diameter [m].
    Cd_fixed : Optional[float]
        If provided (>0), overrides Cd (e.g. from CFD/experiments).
    use_spi_compress : bool
        Enable/disable SPI compressible correction.
    spi_n : float | None
        Isentropic exponent for SPI compressible correction.

    Returns
    -------
    CandidateResult
    """
    p1 = p1_bar * 1e5
    p2 = p2_bar * 1e5
    L_over_D = L / D

    # --- Cd + reference properties from the core ---
    Cd_used, Re_char, rho_l, mu_l = estimate_Cd_from_geometry(
        fluid=fluid,
        p1=p1,
        T_line=T_line,
        D_orif=D,
        L=L,
        r_over_D=r_over_D,
        D_pipe=D_pipe,
        mdot_target=mdot_target,
        Cd_input=Cd_fixed,   # None => geometric estimate, >0 => user Cd
    )

    # --- Phase-aware mass flow + outlet state ---
    mdot_per_hole, model_used, info = solve_mdot_phaseaware(
        fluid=fluid,
        p1=p1,
        p2=p2,
        T_line=T_line,
        D=D,
        Cd=Cd_used,
        use_spi_compress=use_spi_compress,
        spi_n=spi_n,
        L_over_D=L_over_D,
    )

    phase_out = info.get("phase_from_spi", "unknown")

    # Mixture density and viscosity for effective Re
    rho_mix = info.get("rho_out_spi", rho_l)
    mu_mix  = info.get("mu_mix", mu_l)

    A_hole = math.pi * (D**2) / 4.0

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
    """Wrapper per parallelizzazione (ProcessPoolExecutor)."""
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
    use_spi_compress: bool = True,
    spi_n: float | None = None,
) -> List[CandidateResult]:
    """
    3D grid search in (r/D, D, L/D) to find geometries
    that match the target total mass flow.

    Parameters
    ----------
    fluid, p1_bar, p2_bar, T_line : see main().
    mdot_target : float
        TOTAL target mass flow [kg/s].
    nholes : int
        Number of identical holes.
    rD_min, rD_max, n_rD : float, float, int
        Range and number of samples for r/D.
    D_min, D_max, n_D : float, float, int
        Range and number of samples for D [m].
    L_over_D_min, L_over_D_max, n_LD : float, float, int
        Range and number of samples for L/D.
    D_pipe : float
        Upstream pipe diameter [m] (used for β = D_orif / D_pipe).
    n_workers : int
        Number of worker processes for parallel evaluation.
    Cd_fixed : Optional[float]
        If not None, Cd is fixed to this value (bypass geometry model).
    use_spi_compress : bool
        Enable/disable SPI compressible correction in the backend.
    spi_n : float | None
        Isentropic exponent for SPI (if None, backend default is used).

    Returns
    -------
    results : List[CandidateResult]
        List with all evaluated candidates.
    """
    # Grid discretisation
    D_values  = np.linspace(D_min, D_max, n_D)
    LD_values = np.linspace(L_over_D_min, L_over_D_max, n_LD)
    rD_values = np.linspace(rD_min, rD_max, n_rD) if n_rD > 1 else np.array([rD_min])

    # Build parameter list for each grid point
    param_list = []
    for r_over_D in rD_values:
        for D in D_values:
            for LD in LD_values:
                L = LD * D
                param_list.append((
                    fluid,               # 0
                    p1_bar,              # 1
                    p2_bar,              # 2
                    T_line,              # 3
                    mdot_target,         # 4
                    nholes,              # 5
                    float(r_over_D),     # 6
                    float(D),            # 7
                    float(L),            # 8
                    float(D_pipe),       # 9
                    Cd_fixed,            # 10
                    use_spi_compress,    # 11
                    spi_n,               # 12
                ))

    results: List[CandidateResult] = []

    if n_workers is None or n_workers <= 1:
        # Serial execution
        for params in param_list:
            try:
                res = _eval_candidate_wrapper(params)
            except Exception as e:
                # fallback "empty" result with error note
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
        # Parallel execution with processes (use multiple CPU cores)
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


# ================== PLOT RISULTATI ==================
def plot_cd_vs_ratio_by_diameter(
    results: List[CandidateResult],
    mdot_target: float,
    tol_rel_perc: float,
    ax=None,
) -> None:
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

    Se `ax` è None viene creata una nuova figura (comportamento originale).
    Se `ax` è fornito, il grafico viene disegnato su quell'Axes senza aprire
    una nuova finestra.
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

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Diametri distinti (in m)
    unique_D = sorted({round(c.D, 9) for c in feas})

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
            f"L/D = {cand.L_over_D:.4f}\n"
            f"r/D = {cand.r_over_D:.4f}\n"
            f"Re = {cand.Re:.3e}\n"
            f"Cd = {cand.Cd:.3f}\n"
            f"mdot_ratio = {cand.mdot_total / mdot_target:.4f}"
        )
        sel.annotation.set_text(text)
        # Sfondo bianco (non più giallo)
        bbox = sel.annotation.get_bbox_patch()
        bbox.set(fc="white", ec="black", alpha=0.9)

    if created_fig:
        import matplotlib.pyplot as plt
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

def run_gui():
    """
    Tkinter GUI for injector design.

    - Finestra di input (Tkinter) con tutti i parametri e i ToolTip.
    - Alla pressione di "Run design":
        * esegue design_from_mdot(...)
        * apre una finestra matplotlib con:
              - tabella INPUT (sinistra)
              - tabella BEST CANDIDATES (destra)
        * chiama plot_cd_vs_ratio_by_diameter(...) che genera il grafico
          in basso a destra.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # -----------------------------
    #  Simple tooltip for widgets
    # -----------------------------
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

        # -----------------------------------------------------
    #  Output figure con tabelle + grafico
    # -----------------------------------------------------
    def show_output_window(_root,
                           input_params: dict[str, str],
                           candidates: List[CandidateResult],
                           mdot_target: float,
                           tol_rel_perc: float,
                           results_all: List[CandidateResult]) -> None:
        """
        Crea una figura matplotlib con:

          - tabella INPUT (in alto a sinistra)
          - tabella BEST CANDIDATES (in alto a destra)
          - grafico Cd vs mdot_total/mdot_target (in basso a destra).

        Le dimensioni di font e scale delle tabelle vengono adattate
        automaticamente in base a numero di righe/colonne, per usare
        al meglio lo spazio disponibile.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # ----------- Tabella INPUT ----------- 
        ordered_keys = [
            "Fluid",
            "P1 [bar]",
            "P2 [bar]",
            "T_line [K]",
            "mdot target total [kg/s]",
            "Nh [-]",
            "r/D min",
            "r/D max",
            "n_rD",
            "D_min [mm]",
            "D_max [mm]",
            "nD",
            "D_pipe [mm]",
            "L/D min",
            "L/D max",
            "n_LD",
            "tol. rel. [%]",
            "n_workers",
            "Cd fixed",
            "SPI compressible",
            "n SPI",
        ]

        def format_input_value(raw: str) -> str:
            """
            Porta il valore a ~3 cifre decimali (fisso o scientifico).
            Se non è numerico, restituisce la stringa originale.
            """
            s = raw.strip()
            try:
                x = float(s)
            except Exception:
                return raw

            if not math.isfinite(x):
                return raw

            ax = abs(x)
            if (ax != 0.0) and (ax < 1e-3 or ax >= 1e4 or "e" in s.lower()):
                return f"{x:.3e}"
            else:
                return f"{x:.3f}"

        header_input = ["Parameter", "Value"]
        data_input = [header_input]
        for k in ordered_keys:
            if k in input_params:
                raw_val = input_params[k]
                data_input.append([k, format_input_value(raw_val)])

        # ----------- Tabella BEST CANDIDATES ----------- 
        metric_labels = [
            "#",
            "D [mm]",
            "L [mm]",
            "L/D [-]",
            "r/D [-]",
            "Re [-]",
            "Cd [-]",
            "A_h [mm^2]",
            "A_tot [mm^2]",
            "Nh [-]",
            "mdot_hole [kg/s]",
            "mdot_tot [kg/s]",
            "err_rel [%]",
        ]

        def format_best_value(label: str, val: float) -> str:
            """Formattazione numerica per la tabella best."""
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                return "NaN"
            if label == "Re [-]":
                if val == 0.0:
                    return "0.000*10^0"
                exp = int(math.floor(math.log10(abs(val))))
                mant = val / (10.0 ** exp)
                return f"{mant:.3f}*10^{exp}"
            return f"{val:.3f}"

        data_best = [metric_labels]
        for i, c in enumerate(candidates, start=1):
            row = [str(i)]
            for label in metric_labels[1:]:
                if label == "D [mm]":
                    val = c.D * 1e3
                elif label == "L [mm]":
                    val = c.L * 1e3
                elif label == "L/D [-]":
                    val = c.L_over_D
                elif label == "r/D [-]":
                    val = c.r_over_D
                elif label == "Re [-]":
                    val = c.Re
                elif label == "Cd [-]":
                    val = c.Cd
                elif label == "A_h [mm^2]":
                    D_mm = c.D * 1e3
                    val = math.pi * (D_mm ** 2) / 4.0
                elif label == "A_tot [mm^2]":
                    D_mm = c.D * 1e3
                    A_h_mm2 = math.pi * (D_mm ** 2) / 4.0
                    val = A_h_mm2 * c.nh
                elif label == "Nh [-]":
                    val = c.nh
                elif label == "mdot_hole [kg/s]":
                    val = c.mdot_per_hole
                elif label == "mdot_tot [kg/s]":
                    val = c.mdot_total
                elif label == "err_rel [%]":
                    if mdot_target > 0.0:
                        val = abs(c.mdot_total - mdot_target) / mdot_target * 100.0
                    else:
                        val = float("nan")
                else:
                    val = float("nan")

                if label == "Nh [-]":
                    sval = str(int(val))
                else:
                    sval = format_best_value(label, val)
                row.append(sval)
            data_best.append(row)

                # ----------- Figura: 2 righe x 2 colonne ----------- 
        fig_w = 14.0
        fig_h = 9.0

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.canvas.manager.set_window_title("PyInjection – Design results")

        # Colonna sinistra più larga, destra comunque prevalente
        # (circa 30% / 70% della larghezza totale)
        gs = GridSpec(
            2, 2,
            height_ratios=[0.9, 1.6],
            width_ratios=[0.9, 2.1],
            figure=fig,
        )

        ax_input = fig.add_subplot(gs[0, 0])
        ax_best  = fig.add_subplot(gs[0, 1])
        ax_plot  = fig.add_subplot(gs[1, 1])
        ax_bl    = fig.add_subplot(gs[1, 0])

        ax_input.axis("off")
        ax_best.axis("off")
        ax_bl.axis("off")

        ax_input.set_title(
            "Input parameters",
            fontsize=11,
            weight="bold",
            pad=8,
        )
        ax_best.set_title(
            f"Best candidates – target mdot_tot = {mdot_target:.6f} kg/s, "
            f"tolerance = {tol_rel_perc:.3f} %",
            fontsize=11,
            weight="bold",
            pad=8,
        )

        # --- parametri per adattività ---
        n_rows_input = len(data_input)
        n_cols_input = len(data_input[0]) if data_input else 2
        n_rows_best  = len(data_best)
        n_cols_best  = len(data_best[0]) if data_best else len(metric_labels)

        # Font size input: più righe → font leggermente più piccolo
        if n_rows_input > 22:
            fs_input = 7
        elif n_rows_input > 17:
            fs_input = 8
        else:
            fs_input = 9

        # Font size best: più colonne → font più piccolo
        if n_cols_best > 11:
            fs_best = 8
        else:
            fs_best = 9

        # Scale orizzontale best: molti col → scala più piccola
        # con limite minimo per non rendere il testo illeggibile
        scale_x_best = max(0.7, 1.4 - 0.08 * (n_cols_best - 8))
        scale_y_best = 1.8

        # Tabella input: ora più larga (scala_x > 1)
        tab_input = ax_input.table(
            cellText=data_input,
            loc="upper center",
            cellLoc="center",
        )
        tab_input.auto_set_font_size(False)
        tab_input.set_fontsize(fs_input)
        tab_input.scale(1.2, 1.8)   # più larga rispetto a prima

        for (row, col), cell in tab_input.get_celld().items():
            if row == 0:
                cell.set_facecolor("#CCCCCC")
                cell.set_text_props(weight="bold")

        # Tabella best candidates (con scale adattive)
        tab_best = ax_best.table(
            cellText=data_best,
            loc="upper center",
            cellLoc="center",
        )
        tab_best.auto_set_font_size(False)
        tab_best.set_fontsize(fs_best)
        tab_best.scale(scale_x_best, scale_y_best)

        for (row, col), cell in tab_best.get_celld().items():
            if row == 0 or col == 0:
                cell.set_facecolor("#CCCCCC")
                cell.set_text_props(weight="bold")

        # larghezze adattive per sfruttare lo spazio dell’asse destro
        try:
            tab_best.auto_set_column_width(list(range(n_cols_best)))
        except Exception:
            pass

        # Grafico
        ax_plot.clear()
        plot_cd_vs_ratio_by_diameter(
            results_all,
            mdot_target,
            tol_rel_perc,
            ax=ax_plot,
        )

        gs.update(wspace=0.6, hspace=0.8)
        fig.tight_layout()

        plt.show(block=False)


    # -----------------------------
    #  MAIN INPUT WINDOW (Tk)
    # -----------------------------
    root = tk.Tk()
    root.title("PyInjection – Injector Design GUI")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # ----- text entries -----
    fields = [
        ("Fluid (CoolProp)",         "fluid",        "NitrousOxide"),
        ("P1 [bar]",                 "p1_bar",       "55.0"),
        ("P2 [bar]",                 "p2_bar",       "43.0"),
        ("T_line [K]",               "T_line",       "288.0"),
        ("mdot target total [kg/s]", "mdot_target",  "0.140"),
        ("Nh (number of holes)",     "Nh",           "1"),
        ("r/D min",                  "rD_min",       "0.05"),
        ("r/D max",                  "rD_max",       "0.35"),
        ("n_rD",                     "n_rD",         "7"),
        ("D_min [mm]",               "Dmin_mm",      "0.5"),
        ("D_max [mm]",               "Dmax_mm",      "3.5"),
        ("nD",                       "nD",           "25"),
        ("D_pipe [mm]",              "Dpipe_mm",     "5.0"),
        ("L/D min",                  "LD_min",       "2.0"),
        ("L/D max",                  "LD_max",       "12.0"),
        ("n_LD",                     "nLD",          "10"),
        ("tol. rel. [%]",            "tol_rel_perc", "3.0"),
        ("n_workers",                "n_workers",    "6"),
        ("Cd fixed (blank = auto)",  "Cd_fixed",     ""),
        ("n SPI (optional)",         "spi_n",        "1.2"),
    ]

    entries: dict[str, tk.Entry] = {}
    row = 0

    def add_entry(label_text: str, key: str, default: str) -> tk.Entry:
        nonlocal row
        lab = ttk.Label(main_frame, text=label_text + ":")
        lab.grid(row=row, column=0, sticky="e", padx=5, pady=2)
        ent = ttk.Entry(main_frame, width=20)
        ent.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        if default != "":
            ent.insert(0, default)
        entries[key] = ent
        row += 1
        return ent

    for label_text, key, default in fields:
        add_entry(label_text, key, default)

    # ----- Checkbutton SPI compress -----
    spi_comp_var = tk.BooleanVar(value=True)

    chk_spi = ttk.Checkbutton(
        main_frame,
        text="Use SPI compressible correction",
        variable=spi_comp_var,
    )
    chk_spi.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 8))
    row += 1

    # -----------------------------
    #  ToolTips
    # -----------------------------
    ToolTip(entries["fluid"], "CoolProp fluid name used for property evaluation.")
    ToolTip(entries["T_line"], "Feeding-line bulk temperature [K].")
    ToolTip(entries["p1_bar"], "Upstream/inlet pressure P1 [bar].")
    ToolTip(entries["p2_bar"], "Back pressure P2 [bar].")
    ToolTip(entries["mdot_target"], "Total target mass flow rate [kg/s].")
    ToolTip(entries["Nh"], "Number of identical injector holes Nh.")
    ToolTip(entries["rD_min"], "Minimum edge-radius ratio r/D explored in the grid.")
    ToolTip(entries["rD_max"], "Maximum edge-radius ratio r/D explored in the grid.")
    ToolTip(entries["n_rD"], "Number of samples in the r/D range.")
    ToolTip(entries["Dmin_mm"], "Minimum hole diameter D [mm].")
    ToolTip(entries["Dmax_mm"], "Maximum hole diameter D [mm].")
    ToolTip(entries["nD"], "Number of diameter samples in [D_min, D_max].")
    ToolTip(entries["Dpipe_mm"], "Upstream manifold (pipe) diameter D_pipe [mm].")
    ToolTip(entries["LD_min"], "Minimum aspect ratio L/D.")
    ToolTip(entries["LD_max"], "Maximum aspect ratio L/D.")
    ToolTip(entries["nLD"], "Number of samples in the L/D range.")
    ToolTip(entries["tol_rel_perc"], "Relative mass-flow tolerance [%].")
    ToolTip(entries["n_workers"], "Number of worker processes for the grid evaluation.")
    ToolTip(entries["Cd_fixed"], "If blank, Cd is estimated from geometry.")
    ToolTip(entries["spi_n"], "Isentropic exponent n used in SPI compressible correction.")
    ToolTip(chk_spi, "If checked, SPI uses compressible correction with exponent n.")

    # -----------------------------
    #  Run button callback
    # -----------------------------
    def on_run():
        try:
            fluid        = entries["fluid"].get().strip() or "NitrousOxide"
            p1_bar       = float(entries["p1_bar"].get() or 55.0)
            p2_bar       = float(entries["p2_bar"].get() or 43.0)
            T_line       = float(entries["T_line"].get() or 288.0)
            mdot_target  = float(entries["mdot_target"].get() or 0.140)
            Nh           = int(entries["Nh"].get() or 1)
            Nh           = max(Nh, 1)

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

            # Requested number of workers (can be empty or <= 0)
            n_workers_raw = entries["n_workers"].get().strip()
            if n_workers_raw:
                try:
                    n_workers_req = int(n_workers_raw)
                except ValueError:
                    n_workers_req = None
            else:
                n_workers_req = None

            # Effective number of workers based on CPU cores
            n_workers = _auto_n_workers(n_workers_req)

            Cd_fixed_str = entries["Cd_fixed"].get().strip()
            Cd_fixed     = float(Cd_fixed_str) if Cd_fixed_str else None

            spi_n_str    = entries["spi_n"].get().strip()
            if spi_n_str:
                try:
                    spi_n_val = float(spi_n_str)
                except ValueError:
                    spi_n_val = 1.2
                    entries["spi_n"].delete(0, tk.END)
                    entries["spi_n"].insert(0, "1.2")
            else:
                spi_n_val = None

            use_spi_compress = bool(spi_comp_var.get())

        except ValueError as e:
            messagebox.showerror("Input error", f"Invalid numeric value:\n{e}")
            return

        # Conversione mm -> m
        D_min   = Dmin_mm  * 1e-3
        D_max   = Dmax_mm  * 1e-3
        D_pipe  = Dpipe_mm * 1e-3

        # Esecuzione design
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
                use_spi_compress=use_spi_compress,
                spi_n=spi_n_val,
            )
        except Exception as e:
            messagebox.showerror("Design error", f"Error during design grid evaluation:\n{e}")
            return

        if not results:
            messagebox.showwarning("Design", "No results generated (empty list).")
            return

        best_overall = select_best_candidates(results, mdot_target, topk=10)
        feasible     = get_feasible_candidates(results, mdot_target, tol_rel_perc)

        if feasible:
            cand_to_show = feasible[:10]
            msg = f"Design completed.\nCandidates within tolerance: {len(feasible)}"
        else:
            cand_to_show = best_overall
            msg = (
                "Design completed.\n"
                "No candidate within the mass-flow tolerance; "
                "showing global best candidates."
            )

        # Dizionario parametri di input per la tabella
        input_params = {
            "Fluid": fluid,
            "P1 [bar]": f"{p1_bar:.3f}",
            "P2 [bar]": f"{p2_bar:.3f}",
            "T_line [K]": f"{T_line:.3f}",
            "mdot target total [kg/s]": f"{mdot_target:.6f}",
            "Nh [-]": f"{Nh:d}",
            "r/D min": f"{rD_min:.4f}",
            "r/D max": f"{rD_max:.4f}",
            "n_rD": f"{n_rD:d}",
            "D_min [mm]": f"{Dmin_mm:.4f}",
            "D_max [mm]": f"{Dmax_mm:.4f}",
            "nD": f"{nD:d}",
            "D_pipe [mm]": f"{Dpipe_mm:.4f}",
            "L/D min": f"{LD_min:.3f}",
            "L/D max": f"{LD_max:.3f}",
            "n_LD": f"{nLD:d}",
            "tol. rel. [%]": f"{tol_rel_perc:.3f}",
            "n_workers": f"{n_workers:d}",
            "Cd fixed": ("auto (geom)" if Cd_fixed is None else f"{Cd_fixed:.4f}"),
            "SPI compressible": "ON" if use_spi_compress else "OFF",
            "n SPI": ("default" if spi_n_val is None else f"{spi_n_val:.3f}"),
        }

        show_output_window(root, input_params, cand_to_show,
                           mdot_target, tol_rel_perc, results)

        messagebox.showinfo("Design completed", msg)

    run_button = ttk.Button(main_frame, text="Run design", command=on_run)
    run_button.grid(row=row, column=0, columnspan=2, pady=6, sticky="ew")

    main_frame.columnconfigure(1, weight=1)
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

    # r/D range
    parser.add_argument("--rD-min", type=float, default=0.05,
                        help="Minimum r/D. Default: 0.05")
    parser.add_argument("--rD-max", type=float, default=0.35,
                        help="Maximum r/D. Default: 0.35")
    parser.add_argument("--n-rD",   type=int,   default=7,
                        help="Number of r/D samples. Default: 7")

    # D range (hole diameter)
    parser.add_argument("--Dmin-mm", type=float, default=0.5,
                        help="Minimum hole diameter [mm]. Default: 0.5")
    parser.add_argument("--Dmax-mm", type=float, default=3.5,
                        help="Maximum hole diameter [mm]. Default: 3.5")
    parser.add_argument("--nD", type=int, default=25,
                        help="Number of D samples in [Dmin,Dmax]. Default: 25")

    # Upstream pipe diameter
    parser.add_argument("--Dpipe-mm", type=float, default=5.0,
                        help="Manifold (upstream pipe) diameter [mm] for β and RHG. Default: 5.0")

    # L/D range
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

    # Compressibility correction (design side)
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable SPI compressibility correction in the backend.")
    parser.add_argument("--spi-n", type=float, default=None,
                        help="Isentropic exponent n for SPI (optional, e.g. 1.2).")

    parser.add_argument("--gui", action="store_true",
                        help="Launch a simple Tkinter GUI for input.")

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

    tol_rel_perc   = args.tol_rel_perc
    n_workers_req  = args.n_workers
    n_workers      = _auto_n_workers(n_workers_req)
    Cd_fixed       = args.Cd_fixed

    use_spi_compress = not args.no_compress
    spi_n_val        = args.spi_n

    # If requested, launch only the GUI and return
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
    print(f"SPI compress : {'ON' if use_spi_compress else 'OFF'}")
    if spi_n_val is not None:
        print(f"n SPI        : {spi_n_val:.3f}")
    else:
        print("n SPI        : default (backend)")
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
        use_spi_compress=use_spi_compress,
        spi_n=spi_n_val,
    )

    if not results:
        print("No results generated (empty list).")
        return

    best_overall = select_best_candidates(results, mdot_target, topk=args.topk)
    feasible = get_feasible_candidates(results, mdot_target, tol_rel_perc)

    print("=== Selected design candidates ===")
    if feasible:
        to_print = feasible[:args.topk]
        print_candidate_table(to_print, mdot_target)
        if len(feasible) > args.topk:
            print(f"... and {len(feasible) - args.topk} more points satisfy the requirement.\n")
    else:
        print("(no candidate satisfies the mass-flow tolerance; printing global best)\n")
        print_candidate_table(best_overall, mdot_target)

    if not args.no_plot:
        plot_cd_vs_ratio_by_diameter(results, mdot_target, tol_rel_perc)

if __name__ == "__main__":
    import sys

    # Se l'utente lancia il file SENZA argomenti => apri direttamente la GUI
    if len(sys.argv) == 1:
        run_gui()
    else:
        main()