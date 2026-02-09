# Nozzle

This folder contains all the models, data, and numerical tools related
to the design, analysis, and validation of the rocket nozzle.

The workflow follows a progressive approach, starting from analytical
and semi-analytical models, moving through geometric definition, and
finally reaching high-fidelity CFD simulations.

## Project workflow

1. Analytical and system-level models are used to describe the internal
   ballistics of the hybrid rocket motor and to generate time-dependent
   operating conditions.

2. The nozzle baseline geometry is defined using the Method of
   Characteristics (MOC) under on-design conditions.

3. Time-dependent chamber pressure and temperature profiles are used
   to define transient boundary conditions for CFD simulations.

4. CFD analyses are performed to validate nozzle performance and flow
   behavior under both steady and transient operating conditions.

## Folder structure

- MOC  
  Nozzle geometry definition based on the Method of Characteristics.

- AnalyticalModels  
  Analytical and semi-analytical models used to compute internal
  ballistics, performance estimates, and mission profiles.

- BoundaryConditions  
  User-defined functions and data files used to impose time-dependent
  boundary conditions in CFD simulations.

- CFD  
  High-fidelity CFD simulations (2D and 3D), including on-design,
  off-design, optimization and transient analyses.
