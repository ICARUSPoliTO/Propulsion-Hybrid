import numpy as np
from ambiance import Atmosphere


# ============================================================
# Atmosfera
# ============================================================

def ambient_pressure(h_m: float) -> float:
    """
    Ambient static pressure [Pa] at altitude h_m [m]
    using ISA standard atmosphere.
    """
    return float(Atmosphere(h_m).pressure[0])


# ============================================================
# Isentropic relations
# ============================================================

def critical_pressure_ratio(gamma: float) -> float:
    """
    Critical pressure ratio p*/p0 for choking.
    """
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def mach_from_pressure_ratio(gamma: float, pe_pc: float) -> float:
    """
    Exit Mach number from pe/pc (isentropic).
    Valid for pe/pc < p*/p0 (choked flow).
    """
    return np.sqrt(
        (2.0 / (gamma - 1.0))
        * (pe_pc ** (-(gamma - 1.0) / gamma) - 1.0)
    )


def expansion_ratio_from_mach(gamma: float, M: float) -> float:
    """
    Area ratio Ae/At from Mach number (isentropic).
    """
    term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M**2)
    return (1.0 / M) * term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


# ============================================================
# Nozzle sizing (main routine)
# ============================================================

def nozzle_expansion_ratio(
    gamma: float,
    pc_bar: float,
    altitude_m: float,
    pe_bar: float | None = None,
):
    """
    Computes exit Mach number and expansion ratio Ae/At
    for an isentropic, adapted nozzle.

    Parameters
    ----------
    gamma : float
        Specific heat ratio of exhaust gas
    pc_bar : float
        Chamber stagnation pressure [bar]
    altitude_m : float
        Altitude [m]
    pe_bar : float, optional
        Exit pressure override [bar]
        (if None, pe = ambient pressure)

    Returns
    -------
    dict with Me, eps, pe/pc
    """

    pc = pc_bar * 1e5
    pe = pe_bar * 1e5 if pe_bar is not None else ambient_pressure(altitude_m)

    pe_pc = pe / pc
    pe_pc_crit = critical_pressure_ratio(gamma)

    if pe_pc >= pe_pc_crit:
        raise ValueError(
            "Flow not choked: pe/pc >= critical pressure ratio"
        )

    Me = mach_from_pressure_ratio(gamma, pe_pc)
    eps = expansion_ratio_from_mach(gamma, Me)

    return {
        "pe_pc": pe_pc,
        "Me": Me,
        "expansion_ratio": eps,
        "pe_bar": pe / 1e5,
    }


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    gamma = 1.26
    pc_bar = 30.0
    altitude_m = 100.0

    result = nozzle_expansion_ratio(
        gamma=gamma,
        pc_bar=pc_bar,
        altitude_m=altitude_m,
    )

    print("---- Nozzle sizing (isentropic) ----")
    print(f"gamma        = {gamma:.3f}")
    print(f"pc           = {pc_bar:.2f} bar")
    print(f"pe           = {result['pe_bar']:.3f} bar")
    print(f"pe/pc        = {result['pe_pc']:.5f}")
    print(f"Me           = {result['Me']:.3f}")
    print(f"Ae/At (eps)  = {result['expansion_ratio']:.3f}")
