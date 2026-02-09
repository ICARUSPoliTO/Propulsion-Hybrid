#!/usr/bin/env python3
# =============================================================================
# REQUIREMENTS PYTHON NECESSARI PER QUESTO SCRIPT
# =============================================================================
#
# Questo script richiede i seguenti moduli Python:
#
# Standard (già inclusi in Python):
#   - os
#   - sys
#   - math
#   - multiprocessing
#   - csv
#
# Non-standard (da installare con pip):
#   - numpy
#   - scipy
#   - pandas
#   - plotly
#
# Opzionali:
#   - rocketcea     (per integrazione NASA CEA, se disponibile)
#   - pycea         (alternativa a rocketcea)
#
# Installazione tramite terminale:
#
#   pip install numpy
#   pip install scipy
#   pip install pandas
#   pip install plotly
#
# Per CEA (solo se desiderato):
#   pip install rocketcea
#
# Nota: assicurarsi di usare l'interprete Python corretto.
#       Se si usa "python3", sostituire "pip" con "pip3":
#
# =============================================================================
"""
================================================================================
   MODELLO 1D QUASI–STAZIONARIO PER LE PRESTAZIONI DI UN UGELLO DI MOTORE IBRIDO
================================================================================

Questo programma calcola le prestazioni di un ugello di un motore ibrido 
(N2O + paraffina) modellato come flusso comprimibile quasi-steady, quasi-one-dimensional 
con equazioni **isentropiche** e correzioni spannometriche per perdite reali 
(efficienza di efflusso, shock, separazione, ecc.).

Include:
- Calcolo delle grandezze lungo il tempo (0 → t_final).
- Possibile integrazione con NASA CEA (via libreria rocketcea/pycea).
- Possibilità di utilizzare profili sperimentali Pc(t), Tc(t) da file CSV.
- Modello automatico per stimare l’efficienza di efflusso alla gola (eff_choke)
  usando Reynolds, rugosità e spessore di strato limite.
- Output grafico completo e file CSV dei risultati.

===============================================================================
INPUT NECESSARI
===============================================================================

──────────────────────────────────────────────────────────────────────────────
1) GEOMETRIA UGELLO  (necessaria sempre)
──────────────────────────────────────────────────────────────────────────────
- At                : area della gola [m²]
- Ae                : area dell’uscita [m²]
- eff_choke         : coefficiente di efflusso alla gola (0.9–1.0 tipicamente)
- eff_nozzle        : efficienza dell’ugello (per perdita in velocità)
- Cd_surface        : coefficiente per imperfezioni geometriche superficiali
- shock_loss_frac   : frazione perdita spinta dovuta a shock
- sep_loss_frac     : frazione perdita spinta dovuta a separazione del flusso

──────────────────────────────────────────────────────────────────────────────
2) TERMODINAMICA CAMERA DI COMBUSTIONE
──────────────────────────────────────────────────────────────────────────────
A) Se NON si usa NASA CEA:
- Tc_guess          : temperatura stagnazione [K]
- gamma_guess       : γ = cp/cv del gas combusto
- M_guess           : peso molecolare del gas [kg/kmol]

B) Se si usa NASA CEA (in automatico):
- oxidizer          : nome specie ossidante (es. "N2O")
- fuel              : nome specie combustibile (es. "Paraffin" o formula chimica)
- of_ratio          : rapporto O/F in massa
  → Il programma calcolerà automaticamente: Tc(t), gamma(t), M_mw(t)

──────────────────────────────────────────────────────────────────────────────
3) PRESSIONE E TEMPERATURA NEL TEMPO  (scegliere una delle due modalità)
──────────────────────────────────────────────────────────────────────────────
A) Da file CSV (profilo sperimentale):
   Il file deve contenere almeno:
   - t   [s]
   - Pc  [Pa]
   - Tc  [K]    (facoltativa; se mancante si usa Tc_guess o CEA)

B) Generate dal programma:
   Pressione:
   - costante: Pc0
   - lineare:  Pc0, Pc_end
   - esponenziale: Pc0, Pc_end

   Temperatura:
   - costante: Tc_guess
   - oppure calcolata da NASA CEA

──────────────────────────────────────────────────────────────────────────────
4) PARAMETRI AMBIENTE E TEMPORIZZAZIONE
──────────────────────────────────────────────────────────────────────────────
- pa        : pressione ambiente [Pa]
- t_final   : tempo di simulazione [s]
- dt        : passo temporale [s]
- pe_mode   : "isentropic" oppure "ambient"
- outdir    : directory di output per grafici & CSV

──────────────────────────────────────────────────────────────────────────────
5) Stima automatica dell’efficienza di gola (eff_choke) (facoltative, abilitate via interfaccia CLI)
──────────────────────────────────────────────────────────────────────────────
- throat_radius  : raggio geometrico gola [m]
- roughness      : rugosità superficiale media [m]
- x_char         : lunghezza caratteristica per sviluppo BL [m]
- regime         : 'laminar' / 'turbulent'
- valori medi: Pc_med, Tc_med, gamma_med, M_mw, At

Il modello stima:
eff_choke ≈ 1 − k1*(δ/rt) − k2*√(ε_rel);





===============================================================================
OUTPUT DEL PROGRAMMA
===============================================================================

──────────────────────────────────────────────────────────────────────────────
1) File CSV: nozzle_time_history_with_losses.csv
──────────────────────────────────────────────────────────────────────────────
Contiene, per ogni step temporale:
- t              : tempo
- Pc, Tc, gamma, M_mw
- mdot           : portata massica
- Me             : Mach uscita
- Te             : temperatura uscita
- pe             : pressione uscita
- v_e            : velocità uscita
- thrust         : spinta [N]
- Isp            : impulso specifico [s]
- cstar          : velocità caratteristica [m/s]
- Cf             : coefficiente di spinta
- Ae_eff         : area effettiva uscita

──────────────────────────────────────────────────────────────────────────────
2) Grafici in PNG nella cartella di output:
──────────────────────────────────────────────────────────────────────────────
- thrust_vs_time.png
- mdot_vs_time.png
- Isp_vs_time.png
- cstar_vs_time.png
- Mach_vs_time.png
- pe_vs_time.png

──────────────────────────────────────────────────────────────────────────────
3) Se abilitato MoC:
──────────────────────────────────────────────────────────────────────────────
- moc_profile.csv  (x, r, A(x), Mach_est)

===============================================================================
NOTE SUL MODELLO (IMPORTANTE)
===============================================================================
Il modello è **1D quasi-steady**:
- deriva dalla teoria del flusso comprimibile 1D con area variabile,
- assume comportamento isentropico ideale,
- applica fattori correttivi empirici per perdite reali (eff_choke, eff_nozzle,
  shock_loss_frac, sep_loss_frac),
- NON INCLUDE viscosità, turbolenza, swirl, effetti 3D reali,
- non risolve shock complessi né interazione BL–shock.
è un approccio semplificato, per questi ulteriori aspetti è necessaria una CFD.

Per analisi 2D/3D realistiche → serve CFD.

===============================================================================
"""


import os
import sys
import math
import numpy as np
import pandas as pd
from math import isfinite
from scipy.optimize import brentq

# multiprocessing
import multiprocessing as mp
from multiprocessing import cpu_count

# Try to import Plotly for interactive plots; if not installed, inform user.
try:
    import plotly.graph_objects as go
except Exception as e:
    raise ImportError("Plotly non è installato. Installalo con `pip install plotly` per usare i grafici interattivi.") from e

# Optional CEA backend
cea_backend = None
try:
    from rocketcea.cea_obj import CEA_Obj
    cea_backend = 'rocketcea'
except Exception:
    try:
        import pycea
        cea_backend = 'pycea'
    except Exception:
        cea_backend = None

# costanti
R_u = 8314.4621    # J/(kmol K)
g0 = 9.80665       # m/s^2
EPS = 1e-12

# -----------------------------
# Funzioni di utilità (1D)
# -----------------------------
def mach_from_area_ratio(eps, gamma, supersonic=True):
    """
    Risolve per il Mach (M) dato il rapporto area A/A* = eps (Ae/At).
    Uso brentq su equazione isentropica.
    """
    def area_mach_func(M, g, target_eps):
        term = (2.0/(g+1.0))*(1.0 + 0.5*(g-1.0)*M**2)
        rhs = (1.0/M) * term**((g+1.0)/(2.0*(g-1.0)))
        return rhs - target_eps

    if supersonic:
        a = 1.00001
        b = 50.0
    else:
        a = 1e-6
        b = 0.9999

    try:
        fa = area_mach_func(a, gamma, eps)
        fb = area_mach_func(b, gamma, eps)
        if fa * fb > 0:
            return np.nan
        M = brentq(area_mach_func, a, b, args=(gamma, eps), maxiter=200)
        return M
    except Exception:
        return np.nan

# -----------------------------
# Modello di stima eff_choke (opzionale)
# -----------------------------
def sutherland_viscosity(T):
    """
    Legge approssimata della viscosità dinamica (Sutherland), parametri aria-like.
    Restituisce mu [Pa s]
    """
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    mu = mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)
    return mu

def estimate_boundary_layer_thickness(Re_x, x, regime='turbulent'):
    """
    Stima spessore limite delta su piastra:
      - laminar: delta = 5.0 * x / sqrt(Re_x)
      - turbulent: delta = 0.37 * x / Re_x**0.2
    """
    if Re_x <= 0:
        return 0.0
    if regime == 'laminar':
        return 5.0 * x / math.sqrt(Re_x)
    else:
        return 0.37 * x / (Re_x**0.2)

def estimate_eff_choke_from_geometry(Pc, Tc, gamma, M_mw, At, throat_radius,
                                     roughness=1e-5, x_char=None, regime='turbulent'):
    """
    Stima empirica di eff_choke basata su spessore limite e rugosità.
    Restituisce (eff_choke, diagnostics_dict)
    """
    R = R_u / M_mw
    rho0 = Pc / (R * Tc + EPS)
    a_throat = math.sqrt(gamma * R * Tc)
    U = a_throat
    mu = sutherland_viscosity(Tc)
    nu = mu / (rho0 + EPS)
    if x_char is None:
        x_char = 4.0 * throat_radius
    Re_x = U * x_char / (nu + EPS)
    delta = estimate_boundary_layer_thickness(Re_x, x_char, regime=regime)
    circ = 2.0 * math.pi * throat_radius
    area_loss = circ * delta
    frac_area_loss = min(area_loss / (At + EPS), 0.9)
    r_rel = roughness / (throat_radius + EPS)
    k1 = 0.8
    k2 = 0.5
    frac_rough = min(k2 * math.sqrt(r_rel), 0.5)
    eff_choke = 1.0 - k1 * frac_area_loss - frac_rough
    eff_choke = max(0.3, min(1.0, eff_choke))
    diag = {'Re_x': Re_x, 'delta': delta, 'frac_area_loss': frac_area_loss, 'frac_rough': frac_rough}
    return eff_choke, diag

# -----------------------------
# Funzione core per singolo step (calcolo 1D)
# -----------------------------
def nozzle_step(Pc, Tc, gamma, M_mw, At, e0, pa, pe_mode,
                eff_nozzle, eff_choke, Cd_surface, shock_loss_frac, sep_loss_frac):
    """
    Calcolo per singolo istante (scalar inputs).
    Restituisce dizionario con risultati scalari.
    """
    Ae= e0 * At
    R_spec = R_u / (M_mw + EPS)
    Ae_eff = Ae * Cd_surface
    eps = Ae_eff / At

    # choke factor (da teoria isentropica)
    choke_factor = (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))

    # portata ideale choked (kg/s)
    mdot_ideal = At * Pc * math.sqrt(gamma / (R_spec * Tc + EPS)) * choke_factor
    # portata reale
    mdot = eff_choke * mdot_ideal

    # Mach in uscita (ramo supersonico)
    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    if not np.isfinite(Me):
        Me = 2.0  # fallback

    # temperatura statica uscita
    Te = Tc / (1.0 + 0.5 * (gamma - 1.0) * Me**2)

    # pressione statica uscita (isentropica)
    pe_isent = Pc * (Te / Tc)**(gamma / (gamma - 1.0) + EPS)

    if pe_mode == 'ambient':
        pe = pa
    else:
        pe = max(pe_isent, pa)

    # velocità ideale
    v_e_ideal = Me * math.sqrt(gamma * R_spec * Te + EPS)
    # applico efficienze e perdite shock
    v_e = eff_nozzle * v_e_ideal * (1.0 - shock_loss_frac)

    # spinta
    thrust_raw = mdot * v_e + (pe - pa) * Ae_eff
    thrust = thrust_raw * (1.0 - sep_loss_frac)

    # Isp, c*, Cf
    Isp = thrust / (mdot * g0 + EPS)
    cstar = (Pc * At) / (mdot + EPS)
    Cf = thrust / (Pc * At + EPS)

    return {
        'mdot': mdot, 'Me': Me, 'Te': Te, 'pe': pe, 'v_e': v_e,
        'thrust': thrust, 'Isp': Isp, 'cstar': cstar, 'Cf': Cf, 'Ae_eff': Ae_eff
    }

# -----------------------------
# Wrapper per multiprocessing (lavora su indice i)
# -----------------------------
def compute_step_wrapper(args):
    """
    args = (i, Pc_i, Tc_i, gamma_i, M_mw_i, At, Ae, pa, pe_mode,
            eff_nozzle, eff_choke, Cd_surface, shock_loss_frac, sep_loss_frac)
    """
    (i, Pc_i, Tc_i, gamma_i, M_mw_i, At, e0, pa, pe_mode,
     eff_nozzle, eff_choke, Cd_surface, shock_loss_frac, sep_loss_frac) = args
    res = nozzle_step(Pc_i, Tc_i, gamma_i, M_mw_i, At, e0, pa, pe_mode,
                      eff_nozzle, eff_choke, Cd_surface, shock_loss_frac, sep_loss_frac)
    # includo tempo indice per ricostruzione ordinata dopo map
    res['i'] = i
    return res

# -----------------------------
# Interattive plots con Plotly
# -----------------------------
def interactive_plots_from_df(df):
    t = df['t'].values

    def show_series(y, yname, yunit):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=yname))
        fig.update_layout(title=f"{yname} vs time", xaxis_title="t [s]", yaxis_title=f"{yname} ({yunit})",
                          template="plotly_white")
        fig.show()

    show_series(df['thrust'].values, "Thrust", "N")
    show_series(df['mdot'].values, "Mass flow", "kg/s")
    show_series(df['Isp'].values, "Isp", "s")
    show_series(df['cstar'].values, "c*", "m/s")
    show_series(df['Me'].values, "Mach_exit", "-")
    show_series(df['pe'].values, "Exit Pressure", "Pa")

# -----------------------------
# I/O: gestione CSV input
# -----------------------------
def read_time_series_csv(path):
    df = pd.read_csv(path)
    if 't' not in df.columns:
        raise ValueError("CSV deve contenere colonna 't' (tempo).")
    t = df['t'].values.astype(float)
    Pc = df['Pc'].values.astype(float) if 'Pc' in df.columns else None
    Tc = df['Tc'].values.astype(float) if 'Tc' in df.columns else None
    return {'t': t, 'Pc': Pc, 'Tc': Tc}

def build_time_arrays(user_inputs):
    t_final = user_inputs['t_final']
    dt = user_inputs['dt']
    t = np.arange(0.0, t_final + 1e-12, dt)

    # Pressure profile
    if user_inputs.get('Pc_csv'):
        data = read_time_series_csv(user_inputs['Pc_csv'])
        if data['Pc'] is None:
            raise ValueError("File CSV fornito non contiene colonna 'Pc'.")
        Pc = np.interp(t, data['t'], data['Pc'])
    else:
        choice = user_inputs['profile_choice']
        Pc0 = user_inputs['Pc0']; Pc_end = user_inputs['Pc_end']
        if choice == 1:
            Pc = np.linspace(Pc0, Pc_end, t.size)
        elif choice == 2:
            tau = 0.4 * t[-1]
            Pc_tmp = Pc_end + (Pc0 - Pc_end) * np.exp(-t / tau)
            Pc = Pc_end + (Pc0 - Pc_end) * (Pc_tmp - Pc_end) / (Pc_tmp[0] - Pc_end + EPS)
        elif choice == 3:
            Pc = np.full_like(t, Pc0)
        else:
            raise ValueError("Profile choice non supportato")

    # Temperature profile
    if user_inputs.get('Tc_csv'):
        dataT = read_time_series_csv(user_inputs['Tc_csv'])
        if dataT['Tc'] is None:
            raise ValueError("File CSV fornito per Tc non contiene colonna 'Tc'.")
        Tc = np.interp(t, dataT['t'], dataT['Tc'])
    else:
        Tc_guess = user_inputs.get('Tc_guess', 2800.0)
        Tc = np.full_like(Pc, Tc_guess, dtype=float)

    return t, Pc, Tc

# -----------------------------
# Interfaccia utente (CLI) - rimane al suo posto (non spostata all'inizio)
# -----------------------------
def query_user_inputs():
    """Interfaccia a riga di comando (estesa)."""
    print("=== Hybrid nozzle performance tool (V1.1) ===")
    At = float(input("Area gola At [m^2]: ") or "1e-3")
    e0 = float(input("Rapporto di espansione: ") or "5e-3")
    pa = float(input("Pressione ambiente pa [Pa] (default 101325): ") or "101325")

    t_final = float(input("Tempo finale t_final [s] (default. 5.0): ") or "5.0")
    dt = float(input("Step temporale dt [s] (default. 0.01): ") or "0.01")

    # Pc input: CSV o profilo
    use_pc_csv = input("Fornisci Pc(t) da CSV? (y/n) [n]: ") or "n"
    Pc_csv = None
    if use_pc_csv.lower().startswith('y'):
        Pc_csv = input("Percorso file CSV contenente t,Pc (Pa) [es. pc_data.csv]: ")
    print("Profili pressione in camera Pc(t): 1) lineare 2) esponenziale 3) costante")
    choice = int(input("Scegli 1/2/3 (default 1): ") or "1")
    Pc0 = float(input("Pressione iniziale Pc0 [Pa] (default 2.5e6): ") or "2.5e6")
    if choice!=3:
        Pc_end = float(input("Pressione finale Pc_end [Pa] (default 1.5e6): ") or "1.5e6")
    else:
        Pc_end = Pc0


    # Tc input: CSV o costante
    use_tc_csv = input("Fornisci Tc(t) da CSV? (y/n) [n]: ") or "n"
    Tc_csv = None
    if use_tc_csv.lower().startswith('y'):
        Tc_csv = input("Percorso file CSV contenente t,Tc (K) [es. tc_data.csv]: ")
    Tc_guess = float(input("Temperatura stagnazione Tc [K] (default 2800): ") or "2800")

    # CEA o valori manuali
    use_cea = False
    if cea_backend:
        prompt = input(f"Usare CEA via {cea_backend}? (y/n) [n]: ") or "n"
        use_cea = prompt.lower().startswith('y')
    if use_cea:
        oxidizer = input("Nome ossidante per CEA (default [N2O]): ") or "N2O"
        fuel = input("Nome carburante per CEA (default [C12H26]): ") or "C12H26"
        of_ratio = float(input("Rapporto O/F (massa) iniziale (default 6.0): ") or "6.0")
    else:
        gamma_guess = float(input("gamma (cp/cv) (default 1.22): ") or "1.22")
        M_guess = float(input("Peso molecolare medio M [kg/kmol] (default 22.0): ") or "22.0")
        oxidizer = fuel = None
        of_ratio = None

    pe_mode = input("pe_mode: 'ambient' or 'isentropic'? (default [isentropic]) ") or "isentropic"

    # Perdite (default ragionevoli)
    eff_nozzle = float(input("eff_nozzle (0:1) (default [0.97]): ") or "0.97")

    # user can input eff_choke or ask to estimate
    do_estimate_choke = input("Stimare eff_choke automaticamente da geometria? (y/n) [n]: ") or "n"
    eff_choke = None
    eff_choke_diag = None
    if do_estimate_choke.lower().startswith('y'):
        throat_radius = float(input("Raggio gola (throat radius) [m] (default 0.02): ") or "0.02")
        roughness = float(input("Rugosità media [m] (default 1e-5): ") or "1e-5")
        x_char = input("Lunghezza caratteristica x_char [m] (Enter per default 4*rt): ")
        x_char = float(x_char) if x_char.strip() else None
        regime = input("regime strato limite 'turbulent'/'laminar' [turbulent]: ") or "turbulent"
        Pc_med = float(input("Valore medio p_c [Pa] (default 2e6): ") or "2e6")
        Tc_med = float(input("Valore medio Tc [K] (default 2800): ") or "2800")
        gamma_med = float(input("gamma medio (default 1.22): ") or "1.22")
        M_mw_med = float(input("Peso molecolare medio [kg/kmol] (default 22): ") or "22.0")
        At_tmp = float(input("Area gola At [m^2] (default 1e-3): ") or "1e-3")
        eff_choke, eff_choke_diag = estimate_eff_choke_from_geometry(Pc_med, Tc_med, gamma_med, M_mw_med, At_tmp, throat_radius, roughness, x_char, regime)
        print(f"eff_choke stimato = {eff_choke:.4f}; diagnostica: {eff_choke_diag}")
    else:
        eff_choke = float(input("eff_choke (0:1) [0.99]: ") or "0.99")

    Cd_surface = float(input("Cd_surface (0:1) [1.0]: ") or "1.0")
    shock_loss_frac = float(input("shock_loss_frac (0:1) [0.0]: ") or "0.0")
    sep_loss_frac = float(input("sep_loss_frac (0:1) [0.0]: ") or "0.0")

    outdir = input("Cartella output (salva CSV) [./out_nozzle]: ") or "./out_nozzle"
    os.makedirs(outdir, exist_ok=True)

    # CPU parallel options
    auto_cores = input("Usare tutti i core disponibili automaticamente? (y/n) [y]: ") or "y"
    if auto_cores.lower().startswith('y'):
        use_all_cores = True
        ncores = cpu_count()
    else:
        use_all_cores = False
        ncores = int(input(f"Quanti core vuoi usare? (1..{cpu_count()}) [1]: ") or "1")
        ncores = max(1, min(ncores, cpu_count()-1))

    inputs = dict(At=At, e0=e0, pa=pa, t_final=t_final, dt=dt,
                  Pc_csv=Pc_csv, profile_choice=choice, Pc0=Pc0, Pc_end=Pc_end,
                  Tc_csv=Tc_csv, Tc_guess=Tc_guess,
                  use_cea=use_cea, oxidizer=oxidizer, fuel=fuel, of_ratio=of_ratio,
                  pe_mode=pe_mode,
                  eff_nozzle=eff_nozzle, eff_choke=eff_choke,
                  Cd_surface=Cd_surface, shock_loss_frac=shock_loss_frac, sep_loss_frac=sep_loss_frac,
                  outdir=outdir,
                  use_all_cores=use_all_cores, ncores=ncores,
                  eff_choke_diag=eff_choke_diag)
    if not use_cea:
        inputs['gamma_guess'] = gamma_guess
        inputs['M_guess'] = M_guess
    return inputs

# -----------------------------
# MAIN
# -----------------------------
def main():
    inputs = query_user_inputs()
    t, Pc, Tc = build_time_arrays(inputs)
    At = inputs['At']; e0 = inputs['e0']; pa = inputs['pa']

    # ottengo gamma e M (se CEA abilitato, calcolo serialmente PRIMA della parallelizzazione)
    if inputs['use_cea'] and cea_backend:
        print(f"Usando backend CEA: {cea_backend} per calcoli termochimici (calcolo seriale proprietà).")
        C = CEA_Obj(oxName=inputs['oxidizer'], fuelName=inputs['fuel'])
        gamma_list = np.zeros_like(t)
        M_list = np.zeros_like(t)
        Tc_from_cea = np.zeros_like(t)
        for i, Pc_i in enumerate(Pc):
            Pc_bar = Pc_i / 1e5
            try:
                Tc_i = C.get_Temperatures(inputs['of_ratio'], Pc_bar)[0]
            except Exception:
                Tc_i = Tc[i]
            try:
                gam_i = C.get_gamma(inputs['of_ratio'], Pc_bar)
            except Exception:
                gam_i = inputs.get('gamma_guess', 1.22)
            try:
                Mw_i = C.get_Mw(inputs['of_ratio'], Pc_bar)
            except Exception:
                Mw_i = inputs.get('M_guess', 22.0)
            Tc_from_cea[i] = Tc_i
            gamma_list[i] = gam_i
            M_list[i] = Mw_i
        Tc = np.where(Tc_from_cea > 0, Tc_from_cea, Tc)
    else:
        gamma_list = np.full_like(Pc, inputs['gamma_guess'], dtype=float)
        M_list = np.full_like(Pc, inputs['M_guess'], dtype=float)

    # Prepara argomenti per multiprocessing
    N = t.size
    args_list = []
    for i in range(N):
        args_list.append((
            i, float(Pc[i]), float(Tc[i]), float(gamma_list[i]), float(M_list[i]),
            At, e0, inputs['pa'], inputs['pe_mode'],
            inputs['eff_nozzle'], inputs['eff_choke'], inputs['Cd_surface'],
            inputs['shock_loss_frac'], inputs['sep_loss_frac']
        ))

    # Parallel execution (Opzione A: usare ncores)
    ncores = inputs['ncores'] if 'ncores' in inputs else inputs.get('ncores', cpu_count())
    if inputs.get('use_all_cores', True):
        ncores = cpu_count()
    ncores = max(1, min(ncores, cpu_count()))
    print(f"[INFO] Esecuzione su {ncores} core (parallelizzazione).")

    with mp.Pool(processes=ncores) as pool:
        results_list = pool.map(compute_step_wrapper, args_list)

    # ricostruzione risultati ordinati per indice
    results_list_sorted = sorted(results_list, key=lambda x: x['i'])
    mdot = np.array([r['mdot'] for r in results_list_sorted])
    Me = np.array([r['Me'] for r in results_list_sorted])
    Te = np.array([r['Te'] for r in results_list_sorted])
    pe = np.array([r['pe'] for r in results_list_sorted])
    v_e = np.array([r['v_e'] for r in results_list_sorted])
    thrust = np.array([r['thrust'] for r in results_list_sorted])
    Isp = np.array([r['Isp'] for r in results_list_sorted])
    cstar = np.array([r['cstar'] for r in results_list_sorted])
    Cf = np.array([r['Cf'] for r in results_list_sorted])
    Ae_eff = np.array([r['Ae_eff'] for r in results_list_sorted])

    # Salvataggio CSV
    df = pd.DataFrame({
        't': t, 'Pc': Pc, 'Tc': Tc, 'gamma': gamma_list, 'M_mw': M_list,
        'mdot': mdot, 'Me': Me, 'Te': Te, 'pe': pe, 'v_e': v_e,
        'thrust': thrust, 'Isp': Isp, 'cstar': cstar, 'Cf': Cf, 'Ae_eff': Ae_eff
    })
    csvpath = os.path.join(inputs['outdir'], 'nozzle_time_history_with_losses_v2.csv')
    df.to_csv(csvpath, index=False)
    print(f"Risultati salvati: {csvpath}")

    # Grafici interattivi
    print("Apertura grafici interattivi (plotly) nel browser/visualizzatore predefinito...")
    interactive_plots_from_df(df)

if __name__ == "__main__":
    main()

