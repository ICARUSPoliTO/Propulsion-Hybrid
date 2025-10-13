#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, sys
import numpy as np
import CoolProp.CoolProp as cp

def darcy_friction(Re, rel_rough=1e-5):
    """Fattore d'attrito di Darcy:
       - Blasius per Re < 1e5
       - Haaland in generale (robusto)
    """
    if Re < 1e-6:
        return 0.0
    if Re < 1e5:
        return 0.3164 * Re**(-0.25)
    # Haaland:
    return (-1.8*math.log10((rel_rough/3.7)**1.11 + 6.9/Re))**(-2)

def darcy_fully_rough(rel_rough=1e-5):
    """Asintoto pienamente ruvido (indipendente da Re)."""
    return (-1.8*math.log10((rel_rough/3.7)**1.11))**(-2)

def mixture_rho_HEM(fluid, p, h):
    """Ritorna (rho_m, rho_l, rho_v, x, is_two) per HEM con entalpia h costante.
       Se la saturazione a p produce x in [0,1], uso densità omogenea.
       Altrimenti il chiamante gestisce monofase con rho(p,T_line).
    """
    try:
        h_f  = cp.PropsSI('H','P',p,'Q',0, fluid)
        h_g  = cp.PropsSI('H','P',p,'Q',1, fluid)
        rho_l= cp.PropsSI('D','P',p,'Q',0, fluid)
        rho_v= cp.PropsSI('D','P',p,'Q',1, fluid)
        if h_g > h_f:
            x = (h - h_f) / (h_g - h_f)
        else:
            x = -1.0
    except Exception:
        x = -1.0

    if 0.0 <= x <= 1.0:
        rho_m = 1.0 / (x / rho_v + (1.0 - x) / rho_l)
        return rho_m, rho_l, rho_v, x, True
    return None, None, None, 0.0, False

def rho_singlephase_at_T(fluid, p, T):
    return cp.PropsSI('D','P',p,'T',T,fluid)

def solve_mdot_1D(fluid, p1, p2, T_line, D, L, K_minor=0.0, rough=1e-5, n_steps=1000, f_const=None):
    """
    Trova mdot tale che p(z=L) ≈ p2 in una linea 1D adiabatica con h = h1 costante.
    Modello: attrito Darcy–Weisbach su L + perdita concentrata K in uscita.
    - Bifase con HEM (h costante).
    - Monofase isoterma a T_line (rho = rho(p,T_line)).
    - Se le proprietà di trasporto falliscono, fallback pienamente ruvido o f_const.
    Ritorna: dict(mdot, A, G, rho2, U_out).
    """
    A = 0.25*math.pi*D**2
    h1 = cp.PropsSI('H','P',p1,'T',T_line, fluid)

    mdot_lo = 1e-6
    mdot_hi = 5.0
    tol_p   = 500.0
    max_it  = 60

    def outlet_pressure_for_mdot(mdot):
        G = mdot / A
        p = p1
        dz = L / n_steps if n_steps > 0 else L
        for _ in range(n_steps):
            # densità via HEM se bifase, altrimenti monofase a T_line
            rho_m, rho_l, rho_v, x, is_two = mixture_rho_HEM(fluid, p, h1)
            if not is_two:
                rho_m = rho_singlephase_at_T(fluid, p, T_line)

            # fattore d'attrito
            if f_const is not None:
                f = float(f_const)
            else:
                try:
                    if is_two:
                        mu_l = cp.PropsSI('V','P',p,'Q',0,fluid)
                        mu_v = cp.PropsSI('V','P',p,'Q',1,fluid)
                        mu   = (1.0 - x)*mu_l + x*mu_v
                    else:
                        mu   = cp.PropsSI('V','P',p,'T',T_line,fluid)
                    U  = G / rho_m
                    Re = max(rho_m*U*D / mu, 1.0)
                    f  = darcy_friction(Re, rough/D)
                except Exception:
                    f  = darcy_fully_rough(rough/D)

            # dp/dz ≈ -(4f/D) * (G^2 / (2 rho_m))
            dp_dz = - (4.0*f/D) * (G*G) / (2.0*rho_m)
            p += dp_dz * dz
            if p <= 1e3:
                p = 1e3
                break

        # perdita concentrata in uscita
        rho_m_end, *_ = mixture_rho_HEM(fluid, max(p,1e3), h1)
        if rho_m_end is None:
            rho_m_end = rho_singlephase_at_T(fluid, max(p,1e3), T_line)
        p -= K_minor * (G*G) / (2.0*max(rho_m_end,1e-9))
        return p

    p_out_lo = outlet_pressure_for_mdot(mdot_lo)
    p_out_hi = outlet_pressure_for_mdot(mdot_hi)
    if (p_out_lo - p2) * (p_out_hi - p2) > 0:
        for s in [0.1, 10, 50, 100, 200, 500]:
            mdot_hi = s
            p_out_hi = outlet_pressure_for_mdot(mdot_hi)
            if (p_out_lo - p2) * (p_out_hi - p2) <= 0:
                break

    for _ in range(max_it):
        mdot_mid = 0.5*(mdot_lo + mdot_hi)
        p_out_mid = outlet_pressure_for_mdot(mdot_mid)
        if abs(p_out_mid - p2) < tol_p:
            mdot = mdot_mid
            break
        if (p_out_lo - p2)*(p_out_mid - p2) <= 0:
            mdot_hi = mdot_mid
        else:
            mdot_lo = mdot_mid
    else:
        mdot = mdot_mid

    # grandezze finali alla sezione di uscita
    G = mdot / A
    rho2_HEM, *_ = mixture_rho_HEM(fluid, p2, h1)
    if rho2_HEM is None:
        rho2_HEM = rho_singlephase_at_T(fluid, p2, T_line)
    U_out = G / rho2_HEM

    return dict(mdot=mdot, A=A, G=G, rho2=rho2_HEM, U_out=U_out)

# ---------------------- Utilità: velocità nei canalini swirler ----------------------
def swirl_channel_velocities(mdot_total, rho_gas_upstream, rho_liq_at_Tline, d_swirl, n_swirl, use_dpm_flag):
    """
    Calcola le velocità nei canali tangenziali dello swirler dalla PORTATA TOTALE che lo alimenta.
    Ritorna: (u_gas, u_liq, u_selected) [m/s]. (u_selected mantenuto per compatibilità)
    """
    if n_swirl <= 0:
        raise ValueError("n_swirl deve essere >= 1")
    A_swirl = 0.25 * math.pi * d_swirl**2
    mdot_per_channel = mdot_total / n_swirl

    u_gas = mdot_per_channel / max(rho_gas_upstream * A_swirl, 1e-12)
    u_liq = mdot_per_channel / max(rho_liq_at_Tline  * A_swirl, 1e-12)
    u_selected = u_liq if use_dpm_flag else u_gas
    return u_gas, u_liq, u_selected
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    # Ingressi base
    fluid   = "NitrousOxide"
    T_line  = 300.0      # K
    p1_single_bar = 55.0 # <-- imposta None per sweep
    p1_start_bar  = 50.0
    p1_stop_bar   = 60.0
    p1_step_bar   = 1.0

    # Geometria orifizio (1D)
    D       = 0.00547      # m
    L       = 22.68e-3     # m
    K_minor = 1/0.35**2    # perdite concentrate (K = 1/Cd^2), esempio Cd=0.35 da cfd caso comprimibile
    n_holes = 1            # 
    rough   = 1e-5
    f_const = None         # es. 0.02 per bypassare μ/Re

    # Contropressione
    p2_bar  = 43.0

    # Swirler (canali tangenziali)
    d_swirl = 1.70e-3   # [m] diametro di ciascun canale
    n_swirl = 4         # [-] numero di canali tangenziali

    # Costruzione lista p1
    if p1_single_bar is not None:
        p1_list_bar = np.array([float(p1_single_bar)], dtype=float)
        p_desc = f"{p1_single_bar}"
    else:
        p1_list_bar = np.arange(p1_start_bar, p1_stop_bar + p1_step_bar, p1_step_bar, dtype=float)
        p_desc = f"{p1_start_bar}..{p1_stop_bar} (step {p1_step_bar})"

    p2 = p2_bar*1e5

    # Proprietà per un criterio DPM semplificato (solo informativo)
    Tc       = cp.PropsSI('Tcrit', fluid)
    rho_liqT = cp.PropsSI('D','T',T_line,'Q',0,fluid)
    p_vapT   = cp.PropsSI('P','T',T_line,'Q',1,fluid)

    # Array
    N = p1_list_bar.size
    mdot_hole  = np.zeros(N)
    mdot_tot   = np.zeros(N)
    rho_in     = np.zeros(N)
    u_in_gas   = np.zeros(N)
    u_in_liq   = np.zeros(N)
    U_out_hole = np.zeros(N)
    use_DPM    = np.zeros(N, dtype=bool)
    phase_in   = ['']*N

    # Velocità swirler
    u_swirl_gas   = np.zeros(N)
    u_swirl_liq   = np.zeros(N)

    # Aree
    A_c = 0.25*np.pi*D**2
    A_swirl = 0.25*math.pi*d_swirl**2

    # Loop
    for i in range(N):
        p1 = p1_list_bar[i]*1e5

        # Solver 1D per singolo foro
        res = solve_mdot_1D(fluid, p1, p2, T_line, D, L, K_minor, rough=rough, n_steps=800, f_const=f_const)
        mdot_hole[i]  = res["mdot"]
        mdot_tot[i]   = mdot_hole[i] * n_holes
        U_out_hole[i] = res["U_out"]

        # Densità a monte
        rho_in[i] = cp.PropsSI('D','P',p1,'T',T_line,fluid)

        # Velocità medie "d'ingresso" nel foro (ipotesi gas / ipotesi liquido)
        u_in_gas[i] = mdot_hole[i] / (rho_in[i] * A_c)
        u_in_liq[i] = mdot_hole[i] / (rho_liqT   * A_c)

        # Criterio semplice per DPM/flashing (informativo)
        near = 0.02
        is_liq_or_sat = (p1 >= (1.0 - near)*p_vapT)
        will_flash    = (p2 <  (1.0 - near)*p_vapT)
        use_DPM[i]    = bool(is_liq_or_sat and will_flash)

        try:
            phase_in[i] = cp.PhaseSI('P',p1,'T',T_line,fluid)
        except:
            phase_in[i] = 'unknown'

        # Velocità nei canali dello swirler dalla portata totale
        uG, uL, _ = swirl_channel_velocities(
            mdot_total = mdot_tot[i],
            rho_gas_upstream = rho_in[i],
            rho_liq_at_Tline = rho_liqT,
            d_swirl = d_swirl,
            n_swirl = n_swirl,
            use_dpm_flag = use_DPM[i]
        )
        u_swirl_gas[i] = uG
        u_swirl_liq[i] = uL

    # Stima accessoria (facoltativa)
    mfuel = 0.116 * (mdot_tot / (0.25*np.pi*(13.4e-3)**2))**0.331

    # ===================== STAMPA ORDINATA =====================
    print("\n================= INPUT =================")
    print(f"Fluido:                 {fluid}")
    print(f"T_line:                 {T_line:.1f} K   (Tcrit={Tc:.1f} K)")
    print(f"p1 [bar]:               {p_desc}")
    print(f"p2 [bar]:               {p2_bar:.1f}")
    print(f"p_sat(T_line) [bar]:    {p_vapT/1e5:.2f}")
    print("\n--- Orifizio (modello 1D) ---")
    print(f"D = {D*1e3:.2f} mm, L = {L*1e3:.2f} mm, A_c = {A_c:.6e} m^2")
    print(f"n_holes = {n_holes}, K_minor = {K_minor:.2f}, rough = {rough:.2e}, f_const = {f_const}")
    print("\n--- Swirler (canali tangenziali) ---")
    print(f"d_swirl = {d_swirl*1e3:.2f} mm, n_swirl = {n_swirl}, A_swirl (per canale) = {A_swirl:.6e} m^2")

    # --------- TABELLA 1: Orifizio (singolo foro → totale via n_holes) ----------
    print("\n================= TABELLA: ORIFIZIO (PLAIN) =================")
    header_labels_1 = [
        "i","p1 [bar]","n_holes",
        "m_dot_hole [kg/s]","m_dot_tot [kg/s]",
        "rho_in [kg/m^3]",
        "u_in(gas) [m/s]","u_in(liq) [m/s]","U_out(hole) [m/s]",
        "DPM?","fase_in","mfuel [kg/s]"
    ]
    col_widths_1 = [4,10,10,20,20,17,16,16,18,6,12,16]
    header_fmt_1 = "".join([f"{{:<{w}}}" for w in col_widths_1])
    row_fmt_1    = (
        f"{{:<4d}}{{:<10.2f}}{{:<10d}}{{:<20.6e}}{{:<20.6e}}{{:<17.3f}}"
        f"{{:<16.3f}}{{:<16.3f}}{{:<18.3f}}"
        f"{{:<6}}{{:<12}}{{:<16.6e}}"
    )

    print(header_fmt_1.format(*header_labels_1))
    print("-"*sum(col_widths_1))
    for i in range(N):
        print(row_fmt_1.format(
            i+1, p1_list_bar[i], n_holes,
            mdot_hole[i], mdot_tot[i], rho_in[i],
            u_in_gas[i], u_in_liq[i], U_out_hole[i],
            "YES" if use_DPM[i] else "NO", phase_in[i], mfuel[i]
        ))

    # --------- TABELLA 2: Swirler (canali tangenziali) ----------
    print("\n================= TABELLA: SWIRLER (canali tangenziali) =================")
    header_labels_2 = [
        "i","p1 [bar]","m_dot_tot [kg/s]","n_swirl","d_swirl [mm]","A_swirl [m^2]",
        "u_swirl(gas) [m/s]","u_swirl(liq) [m/s]"
    ]
    col_widths_2 = [4,10,20,10,14,16,20,20]
    header_fmt_2 = "".join([f"{{:<{w}}}" for w in col_widths_2])
    row_fmt_2    = (
        f"{{:<4d}}{{:<10.2f}}{{:<20.6e}}{{:<10d}}{{:<14.2f}}{{:<16.6e}}"
        f"{{:<20.3f}}{{:<20.3f}}"
    )

    print(header_fmt_2.format(*header_labels_2))
    print("-"*sum(col_widths_2))
    for i in range(N):
        print(row_fmt_2.format(
            i+1, p1_list_bar[i], mdot_tot[i], n_swirl, d_swirl*1e3, A_swirl,
            u_swirl_gas[i], u_swirl_liq[i]
        ))
