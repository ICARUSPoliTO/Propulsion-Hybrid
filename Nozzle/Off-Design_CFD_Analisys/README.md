# Off-Design Transient CFD Analysis of the Rocket Nozzle

This folder contains the off-design transient CFD analyses of the rocket nozzle, developed as a direct continuation and extension of the on-design CFD campaign.

Unlike the on-design studies, which focus on steady-state operating conditions, the analyses collected here account for time-varying chamber pressure and temperature, following a prescribed mission profile. The objective is to evaluate the nozzle behavior under realistic operating conditions and to assess the robustness of the optimized geometry outside the nominal design point.

This folder is located at the same hierarchical level as the On-Design CFD analyses and follows the same organizational logic for consistency and traceability.



## Scope of the Analysis

The off-design CFD campaign aims to:

- simulate transient nozzle operation driven by realistic chamber pressure and temperature histories;
- capture flow evolution during non-nominal conditions (e.g. start-up, shut-down, intermediate operating points);
- assess internal flow behavior and plume development under varying expansion conditions;
- verify the stability and performance trends of the nozzle geometry outside the design point.



## Analytical Models and Boundary Conditions

The transient inlet boundary conditions are generated using analytical and reduced-order models, which are an integral part of the off-design workflow.

- Chamber pressure and temperature time histories are computed using analytical models based on the mission profile.
- These histories are exported as CSV files and used as inputs for ANSYS Fluent User-Defined Functions (UDFs).
- The UDFs impose time-dependent pressure and temperature at the nozzle inlet during transient CFD simulations.

The analytical models, boundary-condition scripts, and generated data are intentionally kept within the same folder to preserve physical and functional consistency.



## CFD Simulations

The CFD analyses performed in this folder include:

- 2D axisymmetric transient simulations for preliminary verification and rapid assessment;
- 3D transient simulations with an external downstream domain to ensure a physically consistent representation of plume expansion;
- Adaptive Mesh Refinement (AMR) strategies to efficiently capture shocks, expansion fans, and strong gradients during the transient evolution.

Solver settings, turbulence models, and numerical schemes are selected consistently with the on-design campaign, enabling meaningful comparisons.



## Sensitivity and Validation

Where applicable, additional simulations are performed to assess the sensitivity of the off-design results to:

- solver formulation;
- turbulence modeling;
- mesh refinement strategy.

These analyses support the credibility and robustness of the transient CFD results.


## External Data Repository

Due to repository size limitations, the complete CFD cases and post-processing data are stored externally.

Link to full off-design CFD dataset (Drive):  
[INSERT LINK HERE]

The external repository contains:

- full Fluent case and data files;
- mesh files;
- AMR histories;
- post-processing results and figures.

https://politoit-my.sharepoint.com/:f:/g/personal/s309532_studenti_polito_it/IgCVlsMZZcHPSZKlo6R43COjAQDZIdCI1Mc7519WH9lxV1w?e=yoByoU

## Relation to On-Design Analyses

- Same nozzle geometry
- Same CFD framework and solver setup
- Same organizational structure

The off-design campaign complements the on-design analyses by extending the investigation to time-dependent, non-nominal operating conditions, providing a more realistic and complete assessment of nozzle performance.
