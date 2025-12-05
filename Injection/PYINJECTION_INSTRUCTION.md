PYINJECTION - GENERAL INSTRUCTION

PyInjection_core.py contains the flow-rate models (SPI, HEM, NHNE), the outlet-state reconstruction routines, and the geometric discharge-coefficient model Cd(r/D, L/D, Re). It is not meant to be executed directly.
PyInjection_performance.py computes the mass flow rate of an injector with a known geometry by specifying P1, P2, T_line, D, L, r/D and Nh. A discharge coefficient can be estimated from geometry or manually imposed. Pressing Run case displays tables and plots.
PyInjection_design.py explores a grid of geometries (r/D, D, L/D) to match a target total mass flow. The user specifies P1, P2, T_line, the target mass flow, Nh, and the ranges for r/D, D and L/D. A tolerance and an optional fixed Cd can be set, as well as the number of CPU cores via n_workers. Pressing Run design displays the best candidate configurations and the Cd vs flow-ratio plot.

The estimated Cd is based on a simplified geometric correlation and should only be used for preliminary design. It is strongly recommended to validate the chosen geometry through CFD to obtain an accurate Cd to be used in simulations or hardware testing.
Always verify unit consistency in the input parameters.
The SPI/NHNE models are selected automatically by the phase-aware backend.
For dense design grids, increase n_workers to take advantage of available CPU cores.

For further details, consult the dedicated manual.