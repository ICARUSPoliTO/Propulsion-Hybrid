# On-Design CFD Analyses

This folder documents the on-design CFD analyses performed to validate
the optimized nozzle geometry under nominal operating conditions.

Due to the size of the numerical datasets (meshes, case files, and
solution data), the full CFD simulations are not stored directly in
this repository.

## Scope of the analyses

The on-design CFD activity includes:

- High-fidelity 3D simulations of the optimized nozzle geometry
- Inclusion of the external flow domain downstream of the nozzle exit
- Adaptive Mesh Refinement (AMR) based on flow features
- Solver and physical model sensitivity analyses

## Analysis structure

The CFD activity is organized as follows:

- CFD_3D_OnDesign_AMR_ExternalDomain  
  Three-dimensional on-design CFD simulations including the external
  domain and adaptive mesh refinement strategies.

- Solver&Model_Sensibility_Analysis  
  Sensitivity analyses performed to assess the impact of solver choice
  and physical modeling assumptions on the predicted nozzle performance.

## Access to CFD data

The complete CFD datasets and simulation files are available at the
following location:

https://politoit-my.sharepoint.com/:f:/g/personal/s309532_studenti_polito_it/IgAzyQS21RyNSbW2zW4amk4oAc4rkvj4k-5kH-saAKEiqRE?e=UuXLRd

The directory contains the full Fluent cases, meshes, solution files,
and post-processing data.
