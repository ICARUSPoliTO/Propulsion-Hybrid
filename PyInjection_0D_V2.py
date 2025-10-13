import CoolProp.CoolProp as cp 
import numpy as np
import matplotlib.pyplot as plt

class Injector(object):
    def __init__(self, fluid):
        # Check if the fluid is available in CoolProp
        if fluid in cp.FluidsList():
            self.fluid = fluid
        else:
            print("Fluid not found")
            print(cp.FluidsList())
            exit('Read the list above and try again!')

    def injection_area(self, D, n):
        # Compute total injection area (m^2)
        self.A = 0.25 * n * np.pi * (D ** 2)

    def massflow(self, p1, p2, T, cD):
        # Isothermal flow assumption
        h1 = cp.PropsSI('H', 'P', p1, 'T', T, self.fluid)
        h2 = cp.PropsSI('H', 'P', p2, 'T', T, self.fluid)
        d2 = cp.PropsSI('D', 'P', p2, 'T', T, self.fluid)
        dSPI = cp.PropsSI('D', 'T', T, 'Q', 0, self.fluid)
        pV = cp.PropsSI('P', 'T', T, 'Q', 1, self.fluid)
        if p1 > p2:
            # Single-phase incompressible (SPI) and Homogeneous Equilibrium Model (HEM)
            mdot_SPI = cD * np.sqrt(2 * dSPI * (p1 - p2))
            mdot_HEM = cD * d2 * np.sqrt(2 * abs(h1 - h2))
            # Blending coefficient (Waxman & Cantwell model)
            k = np.sqrt((p1 - p2) / (pV - p2))
            mdot = (k * mdot_SPI / (k + 1) + mdot_HEM / (k + 1))
            self.mdot = mdot
        else:
            # No backflow allowed
            self.mdot = 0
            self.mdot_SPI = 0
            self.mdot_HEM = 0

if __name__ == '__main__':
    plt.close('all')

    ox = Injector('NitrousOxide')

    # === Exit orifice (single hole) ===
    D_exit = 0.00547
    ox.injection_area(D_exit, 1)

    # === Pressure setup: single value or range ===
    p_single_bar = 55           # <-- set to None for range
    p_start_bar  = 50
    p_stop_bar   = 60
    p_step_bar   = 1

    if p_single_bar is not None:
        pinj = np.array([float(p_single_bar)], dtype=float)
        p_desc = f"{p_single_bar}"
    else:
        pinj = np.arange(p_start_bar, p_stop_bar + p_step_bar, p_step_bar, dtype=float)
        p_desc = f"{p_start_bar}..{p_stop_bar} (step {p_step_bar})"

    pc      = 43.0       # Chamber pressure [bar]
    T_line  = 300.0      # Feed line temperature [K]
    cD_eff  = 0.33       # Effective discharge coefficient

    # === Swirler inlet channels ===
    D_c = 0.00170        # Channel diameter [m]
    n_c = 4              # Number of channels
    A_c = 0.25 * np.pi * D_c**2

    # === Arrays for results ===
    N  = pinj.size
    mdot_tot   = np.zeros(N)
    mdot_c     = np.zeros(N)
    rho_in     = np.zeros(N)
    u_can_gas  = np.zeros(N)
    u_can_liq  = np.zeros(N)
    use_DPM    = np.zeros(N, dtype=bool)
    phase_in   = ['']*N

    # === Thermodynamic properties ===
    Tc = cp.PropsSI('Tcrit', ox.fluid)
    rho_liq_T = cp.PropsSI('D','T',T_line,'Q',0,ox.fluid)
    p_vap_T   = cp.PropsSI('P','T',T_line,'Q',1,ox.fluid)

    # === Loop over inlet pressures ===
    for i in range(N):
        p_in  = pinj[i] * 1e5
        p_out = pc * 1e5
        ox.massflow(p_in, p_out, T_line, cD_eff)

        mdot_tot[i] = ox.mdot * ox.A
        mdot_c[i]   = mdot_tot[i] / n_c
        rho_in[i]   = cp.PropsSI('D','P',p_in,'T',T_line,ox.fluid)
        u_can_gas[i] = mdot_c[i] / (rho_in[i] * A_c)
        u_can_liq[i] = mdot_c[i] / (rho_liq_T * A_c)

        # Determine if DPM should be used (liquid → flashing)
        near = 0.02
        is_liq_or_sat = (p_in >= (1.0 - near) * p_vap_T)
        will_flash    = (p_out <  (1.0 - near) * p_vap_T)
        use_DPM[i]    = bool(is_liq_or_sat and will_flash)

        try:
            phase_in[i] = cp.PhaseSI('P',p_in,'T',T_line,ox.fluid)
        except:
            phase_in[i] = 'unknown'

    # === Empirical fuel mass flow correlation (for comparison) ===
    mfuel = 0.116 * (mdot_tot / (0.25*np.pi*(13.4E-3)**2))**0.331

    # === PRINT OUTPUT TABLE ===
    print("\n================= INPUT =================")
    print(f"Fluid:                  {ox.fluid}")
    print(f"T (feed line):          {T_line:.1f} K   (Tcrit={Tc:.1f} K)")
    print(f"p_in (bar):             {p_desc}")
    print(f"p_out (bar):            {pc:.1f}")
    print(f"p_vap(T) [bar]:         {p_vap_T/1e5:.2f}")
    print("\n--- Geometry ---")
    print(f"Exit orifice: D_exit = {D_exit*1e3:.2f} mm, A_exit = {ox.A:.6e} m^2")
    print(f"Channels: D_c = {D_c*1e3:.2f} mm, A_c = {A_c:.6e} m^2, n_c = {n_c}")
    print(f"Effective cD:           {cD_eff:.3f}")
    print(f"rho_liq@T_line:         {rho_liq_T:.1f} kg/m^3")

    print("\n================= RESULTS (for each p_in) =================")
    header_labels = [
        "i", "p_in [bar]", "m_dot_tot [kg/s]", "m_dot_c [kg/s]",
        "rho_in [kg/m^3]", "u_can(gas) [m/s]", "u_can(DPM-liq) [m/s]",
        "DPM?", "phase_in", "mfuel [kg/s]"
    ]
    col_widths = [4, 12, 20, 18, 17, 18, 22, 6, 10, 16]
    header_fmt = "".join([f"{{:<{w}}}" for w in col_widths])
    row_fmt    = (
        f"{{:<4d}}{{:<12.2f}}{{:<20.6e}}{{:<18.6e}}{{:<17.3f}}{{:<18.3f}}{{:<22.3f}}"
        f"{{:<6}}{{:<10}}{{:<16.6e}}"
    )

    print(header_fmt.format(*header_labels))
    print("-" * sum(col_widths))
    for i in range(N):
        print(row_fmt.format(
            i+1, pinj[i], mdot_tot[i], mdot_c[i], rho_in[i],
            u_can_gas[i], u_can_liq[i],
            "YES" if use_DPM[i] else "NO",
            phase_in[i], mfuel[i]
        ))
