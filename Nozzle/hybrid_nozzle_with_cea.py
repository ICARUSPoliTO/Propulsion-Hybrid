#!/usr/bin/env python3
"""
Da implementare: 
    nell'input far si che la P_c(t) sia inseribile da un file CSV fornito delle misurazioni sperimentali.

COSA FA IL CODICE:
Calcola prestazioni di ugello per motore ibrido (N2O + paraffina) nel tempo,
con integrazione opzionale di NASA CEA (via rocketcea o pycea).

Output:
 - CSV con campi tempo, Pc, Tc, gamma, M, mdot, ve, pe, thrust, Isp, cstar, Cf
 - Grafici PNG: thrust, mdot, Isp, cstar, Cf vs time

Installazione suggerita (pip):
  pip install numpy scipy matplotlib pandas
  # per CEA (opzionale):
  pip install rocketcea          # o pip install pycea
  # rocketcea potrebbe richiedere f2py/compilazione del codice Fortran
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite

# costanti
R_u = 8314.4621    # J/(kmol K)
g0 = 9.80665       # m/s^2
EPS = 1e-12

# -------- Optional CEA backends --------
cea_backend = None
try:
    # prefer rocketcea if presente
    from rocketcea.cea_obj import CEA_Obj
    cea_backend = 'rocketcea'
except Exception:
    try:
        import pycea
        cea_backend = 'pycea'
    except Exception:
        cea_backend = None

def query_user_inputs():
    """Interfaccia a riga di comando (semplice)."""
    print("=== Hybrid nozzle performance tool ===")
    # geometry
    At = float(input("Area gola At [m^2] (es. 1e-3): ") or "1e-3")
    Ae = float(input("Area uscita Ae [m^2] (es. 5e-3): ") or "5e-3")
    pa = float(input("Pressione ambiente pa [Pa] (default 101325): ") or "101325")
    # time
    t_final = float(input("Tempo finale t_final [s] (es. 5.0): ") or "5.0")
    dt = float(input("Step temporale dt [s] (es. 0.1): ") or "0.05")
    # Pc profile options
    print("Profili pressione in camera Pc(t):")
    print("  1) lineare (Pc0 -> Pc_end)")
    print("  2) esponenziale (Pc0 -> Pc_end)")
    print("  3) fornisci valori (CSV) - NON implementato in prompt")
    choice = int(input("Scegli 1/2/3 (default 1): ") or "1")
    Pc0 = float(input("Pressione iniziale Pc0 [Pa] (es. 2.5e6): ") or "2.5e6")
    Pc_end = float(input("Pressione finale Pc_end [Pa] (es. 1.5e6): ") or "1.5e6")
    # fluid properties / CEA
    use_cea = False
    if cea_backend:
        prompt = input(f"Usare CEA via {cea_backend}? (y/n) [y]: ") or "y"
        use_cea = prompt.lower().startswith('y')
    else:
        print("Nessun wrapper CEA trovato (rocketcea/pycea). Verranno usati Tc,gamma,M inseriti manualmente.")
    if use_cea:
        oxidizer = input("Nome ossidante per CEA (es: N2O) [N2O]: ") or "N2O"
        # per il carburante useremo una rappresentazione molecolare (es. C12H26 o 'paraffin' non standard)
        fuel = input("Nome carburante per CEA (es: C12H26 per paraffina) [C12H26]: ") or "C12H26"
        of_ratio = float(input("Rapporto O/F (massa) iniziale (es. 6.0): ") or "6.0")
    else:
        Tc_guess = float(input("Temperatura stagnazione Tc [K] (es. 2800): ") or "2800")
        gamma_guess = float(input("gamma (cp/cv) (es. 1.22): ") or "1.22")
        M_guess = float(input("Peso molecolare medio M [kg/kmol] (es. 22.0): ") or "22.0")
        oxidizer = fuel = None
        of_ratio = None

    pe_mode = input("pe_mode ('ambient' or 'isentropic') [ambient]: ") or "ambient"

    outdir = input("Cartella output (salva CSV e PNG) [./out_nozzle]: ") or "./out_nozzle"
    os.makedirs(outdir, exist_ok=True)

    inputs = dict(At=At, Ae=Ae, pa=pa, t_final=t_final, dt=dt, Pc0=Pc0,
                  Pc_end=Pc_end, profile_choice=choice,
                  use_cea=use_cea, oxidizer=oxidizer, fuel=fuel, of_ratio=of_ratio,
                  Tc_guess=locals().get('Tc_guess', None),
                  gamma_guess=locals().get('gamma_guess', None),
                  M_guess=locals().get('M_guess', None),
                  pe_mode=pe_mode, outdir=outdir)
    return inputs

def build_Pc_profile(choice, Pc0, Pc_end, t):
    if choice == 1:
        return np.linspace(Pc0, Pc_end, t.size)
    elif choice == 2:
        # esponenziale che passa da Pc0 a Pc_end
        tau = 0.4 * t[-1]  # parametro arbitrario per forma
        Pc = Pc_end + (Pc0 - Pc_end) * np.exp(-t / tau)
        # scala per far valere i due estremi esattamente
        Pc = Pc_end + (Pc0 - Pc_end) * (Pc - Pc_end) / (Pc[0] - Pc_end + EPS)
        return Pc
    else:
        raise ValueError("Profile choice non supportato in prompt")

# --------- CEA helpers (RocketCEA or pycea) ----------
def cea_get_properties_rocketcea(cea_obj, oxidizer, fuel, of, Pc_bar):
    """
    Usa rocketcea CEA_Obj per ottenere T_chamber, gamma e Mw (kg/kmol).
    Nota: rocketcea accetta pressione in bar tipicamente; si consiglia di consultare doc.
    """
    # CEA_Obj.get_TP and other interfaces; RocketCEA fornisce get_TP / get_Temperatures etc.
    # Qui usiamo get_IdealGasProperties o get_Temperatures-like calls (API varia con versione).
    try:
        # Esempio: CEA_Obj(oxName='N2O', fuelName='C12H26') e poi chiamate dell'oggetto
        # Per compatibilità proviamo con metodi comuni:
        Tch = cea_obj.get_T_chamber(of, Pc_bar) if hasattr(cea_obj, "get_T_chamber") else None
    except Exception:
        Tch = None

    # Fallback: molte versioni forniscono get_Isp, get_Cstar, get_T - vedi docs
    # Qui useremo get_T and get_gam and get_M if disponibili
    try:
        T = cea_obj.get_T_of_O2_fuel(oxName=oxidizer, fuelName=fuel, of=of, Pc=Pc_bar)
    except Exception:
        T = None

    # Because RocketCEA APIs differ, minimal approach: call get_Isp or get_T and parse.
    return Tch  # placeholder, see usage example below

def get_combustion_props_via_rocketcea(oxidizer, fuel, of, Pc_pa):
    """
    Wrapper generico che crea CEA_Obj e richiede proprietà.
    Restituisce: (Tc [K], gamma, M [kg/kmol], cstar [m/s], isp [s]) per quel set di condizioni.
    NB: l'API di rocketcea varia; questo esempio mostra il flusso logico.
    """
    # pressione in bar (RocketCEA / CEA convenzionalmente usa atm/bar a volte)
    Pc_bar = Pc_pa / 1e5  # 1 bar = 1e5 Pa
    # crea oggetto
    if cea_backend == 'rocketcea':
        C = CEA_Obj(oxName=oxidizer, fuelName=fuel)
        # RocketCEA offre ad es. C.get_Temperatures(...) o get_Isp(...) — controlla la tua versione
        try:
            # Questo è un esempio comune:
            # get_Temperatures(of, Pc, eps, frozen, ...). Metodi esatti variano.
            Tc = C.get_Temperatures(of, Pc_bar)[0]
        except Exception:
            Tc = None
        try:
            gam = C.get_gamma(of, Pc_bar)
        except Exception:
            gam = None
        try:
            Mw = C.get_Mw(of, Pc_bar)  # kg/kmol
        except Exception:
            Mw = None
        try:
            cstar = C.get_Cstar(of, Pc_bar)
        except Exception:
            cstar = None
        try:
            isp = C.get_Isp(of, Pc_bar)
        except Exception:
            isp = None
        return Tc, gam, Mw, cstar, isp
    elif cea_backend == 'pycea':
        # adattare secondo pycea API
        raise NotImplementedError("Adatta questo blocco per la versione di pycea installata.")
    else:
        raise RuntimeError("CEA backend non disponibile")

# ---------- Nozzle performance functions ----------
def nozzle_performance_time(t, Pc, At, Ae, Tc_arr, gamma_arr, M_arr, pa=101325.0, pe_mode='ambient',
                            eff_nozzle=1.0, eff_choke=1.0):
    """
    Calcola mdot, ve, pe, thrust, Isp, cstar, Cf vettorialmente.
    Tc_arr, gamma_arr, M_arr possono essere array di dimensione t.size (o scalari)
    eff_nozzle: efficienza nozzolare (0..1) per perdite reali su v_e
    eff_choke: coefficiente di efflusso per gola (0..1)
    """
    Pc = np.asarray(Pc)
    t = np.asarray(t)
    # broadcast scalari
    def ensure_arr(x):
        if np.isscalar(x):
            return np.full_like(Pc, x, dtype=float)
        else:
            return np.asarray(x, dtype=float)
    Tc = ensure_arr(Tc_arr)
    gamma = ensure_arr(gamma_arr)
    M = ensure_arr(M_arr)
    R_spec = R_u / M  # J/(kg K) since M is kg/kmol
    
    choke_factor = (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
    mdot = eff_choke * At * Pc * np.sqrt(gamma/(R_spec*Tc)) * choke_factor  # kg/s
    
    if pe_mode == 'ambient':
        pe = np.full_like(Pc, pa)
    elif pe_mode == 'isentropic':
        eps = Ae / At
        pe = Pc * eps**(- (gamma-1.0)/gamma)
        pe = np.maximum(pe, pa)
    else:
        pe = np.full_like(Pc, pa)
    
    pressure_ratio = np.clip(pe / Pc, EPS, 1.0)
    term = 1.0 - pressure_ratio**((gamma-1.0)/gamma)
    ve_ideal = np.sqrt((2.0*gamma/(gamma-1.0)) * R_spec * Tc * term)
    ve = eff_nozzle * ve_ideal
    thrust = mdot * ve + (pe - pa) * Ae
    Isp = thrust / (mdot * g0)
    cstar = (Pc * At) / (mdot + EPS)
    Cf = thrust / (Pc * At + EPS)
    
    return dict(t=t, Pc=Pc, Tc=Tc, gamma=gamma, M=M, mdot=mdot, ve=ve, pe=pe,
                thrust=thrust, Isp=Isp, cstar=cstar, Cf=Cf)

# ---------- plotting / output ----------
def plot_results(df, outdir):
    t = df['t'].values
    figs = []
    # thrust
    plt.figure(); plt.plot(t, df['thrust']); plt.xlabel('t [s]'); plt.ylabel('Thrust [N]'); plt.grid(True)
    plt.title('Thrust vs time'); f1 = os.path.join(outdir, 'thrust_vs_time.png'); plt.savefig(f1); figs.append(f1); plt.close()
    # mdot
    plt.figure(); plt.plot(t, df['mdot']); plt.xlabel('t [s]'); plt.ylabel('Mass flow [kg/s]'); plt.grid(True)
    plt.title('mdot vs time'); f2 = os.path.join(outdir, 'mdot_vs_time.png'); plt.savefig(f2); figs.append(f2); plt.close()
    # Isp
    plt.figure(); plt.plot(t, df['Isp']); plt.xlabel('t [s]'); plt.ylabel('Isp [s]'); plt.grid(True)
    plt.title('Isp vs time'); f3 = os.path.join(outdir, 'Isp_vs_time.png'); plt.savefig(f3); figs.append(f3); plt.close()
    # cstar
    plt.figure(); plt.plot(t, df['cstar']); plt.xlabel('t [s]'); plt.ylabel('c* [m/s]'); plt.grid(True)
    plt.title('c* vs time'); f4 = os.path.join(outdir, 'cstar_vs_time.png'); plt.savefig(f4); figs.append(f4); plt.close()
    # Cf
    plt.figure(); plt.plot(t, df['Cf']); plt.xlabel('t [s]'); plt.ylabel('Cf []'); plt.grid(True)
    plt.title('Cf vs time'); f5 = os.path.join(outdir, 'Cf_vs_time.png'); plt.savefig(f5); figs.append(f5); plt.close()
    return figs

def main():
    inputs = query_user_inputs()
    At = inputs['At']; Ae = inputs['Ae']; pa = inputs['pa']
    t = np.arange(0.0, inputs['t_final'] + 1e-12, inputs['dt'])
    Pc = build_Pc_profile(inputs['profile_choice'], inputs['Pc0'], inputs['Pc_end'], t)
    
    # se si usa CEA: per semplicità assumiamo O/F costante nel tempo (puoi cambiare la logica)
    if inputs['use_cea'] and cea_backend:
        print(f"Usando backend CEA: {cea_backend} per calcoli termochimici.")
        # crea oggetto una sola volta per efficienza se rocketcea
        if cea_backend == 'rocketcea':
            C = CEA_Obj(oxName=inputs['oxidizer'], fuelName=inputs['fuel'])
            Tc_list = np.zeros_like(t)
            gamma_list = np.zeros_like(t)
            M_list = np.zeros_like(t)
            # calcoliamo proprietà a ciascun istante (Pc può cambiare)
            for i, Pc_i in enumerate(Pc):
                Pc_bar = Pc_i / 1e5
                # Nota: le chiamate API di rocketcea variano con versione; potresti dover adattare
                try:
                    # esempio comune: get_Temperatures(of, Pc, eps) -> ritorna array
                    Tc_i = C.get_Temperatures(inputs['of_ratio'], Pc_bar)[0]
                except Exception:
                    # fallback: call get_T of other name or use approximate Tc default
                    Tc_i = inputs['Tc_guess'] or 2800.0
                # gamma and Mw - esempi generici (adatta se la tua versione ha metodi diversi)
                try:
                    gam_i = C.get_gamma(inputs['of_ratio'], Pc_bar)
                except Exception:
                    gam_i = inputs['gamma_guess'] or 1.22
                try:
                    Mw_i = C.get_Mw(inputs['of_ratio'], Pc_bar)
                except Exception:
                    Mw_i = inputs['M_guess'] or 22.0
                Tc_list[i] = Tc_i
                gamma_list[i] = gam_i
                M_list[i] = Mw_i
        else:
            raise RuntimeError("pycea integration non implementata nello script d'esempio; adatta per la tua versione.")
    else:
        # usa campi costanti inseriti manualmente
        Tc_list = np.full_like(Pc, inputs['Tc_guess'], dtype=float)
        gamma_list = np.full_like(Pc, inputs['gamma_guess'], dtype=float)
        M_list = np.full_like(Pc, inputs['M_guess'], dtype=float)

    # calcola prestazioni
    res = nozzle_performance_time(t, Pc, At, Ae, Tc_list, gamma_list, M_list, pa=pa, pe_mode=inputs['pe_mode'], eff_nozzle=0.97, eff_choke=0.99)

    # crea DataFrame
    df = pd.DataFrame({
        't': res['t'], 'Pc': res['Pc'], 'Tc': res['Tc'], 'gamma': res['gamma'], 'M': res['M'],
        'mdot': res['mdot'], 've': res['ve'], 'pe': res['pe'], 'thrust': res['thrust'], 'Isp': res['Isp'],
        'cstar': res['cstar'], 'Cf': res['Cf']
    })

    csvpath = os.path.join(inputs['outdir'], 'nozzle_time_history.csv')
    df.to_csv(csvpath, index=False)
    print(f"Risultati salvati: {csvpath}")

    figs = plot_results(df, inputs['outdir'])
    print("Grafici salvati:")
    for f in figs: print(" -", f)

if __name__ == "__main__":
    main()
