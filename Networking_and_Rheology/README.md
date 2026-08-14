# Evaluation of networking behaviour and rheological properties of DNA nanomotif networks

This folder contains scripts and notebooks for simulations of X-shaped nanomotifs that assemble into networks, which can be evaluated with graph-based rheology. Simulation files evaluated as well as copies of all relevant scripts were uploaded to zenodo (https://doi.org/10.5281/zenodo.18338015). Additionally, a compact, self-contained tutorial that walks through to the most important steps of the full workflow is available in the tutorial folder. The full analysis is split into the following notebooks:

NCG_thermodyn_x_motif.ipynb:
Thermodynamic insights into X-shaped nanomotifs from oxDNA simulations. MD runs and umbrella sampling are evaluated. (Data in oxDNA_simulations.zip on zenodo)

NCG_target_angles_x_motif.ipynb:
Angles within and between arms of X-shaped nanomotifs are derived from oxDNA MD runs. These serve as optimisation targets for coarse-grained bead-spring models. (Data in oxDNA_simulations.zip on zenodo)

NCG_BO_opt_validation.ipynb:
To validate Bayesian parameter optimisation for the bead-spring model, targets from random bead-spring parameters are selected and used for iterative optimisation. (Data in ReaDDy_simulations_opt_eval.zip on zenodo)

NCG_BO_opt_validation_2.ipynb:
Updated validation with finite-size correction for KS-test. (Data in ReaDDy_simulations_opt_eval.zip on zenodo)

NCG_BO_opt_x_motif.ipynb:
Bayesian optimisation of bead-spring parameters, given oxDNA target angle distributions. (Data in ReaDDy_simulations.zip on zenodo)

NCG_BO_opt_x_motif_m2.ipynb
Bayesian optimisation of bead-spring parameters, using combined optimisation of parameters. (Data in ReaDDy_simulations.zip on zenodo)

NCG_network_properties.ipynb:
Evaluation of network properties and of rheological behaviour of nanomotif networks assembled with bead-spring simulations. (Data in ReaDDy_simulations.zip on zenodo)

NCG_network_properties_2.ipynb:
Evaluation of network properties and rheological behaviour including hydrodynamic interactions and 3-armed versions of nanomotif networks assembled with bead-spring simulations. (Data in ReaDDy_simulations.zip on zenodo)

NCG_network_properties_3.ipynb:
Extended analysis of network properties with additional validation, stability and finite-size tests as well as test of graph-based predictions of absolute moduli shifts. (Data in ReaDDy_simulations.zip on zenodo)

NCG_run_networking_sim_1.ipynb:
Supporting script, only to run equilibration and production simulations in ReaDDy.

NCG_ML_rheo_pred.ipynb:
Using Gaussian processes and Bayesian optimisation, connectivity between nanomotifs is predicted given target rheological responses. (Data in ReaDDy_simulations.zip on zenodo)

NCG_SE_seq_prediction.ipynb:
NUPACK is used in combination with Bayesian optimisation to predict which sticky ends sequences can realise a target connectivity in nanomotif networks. (Data in ReaDDy_simulations.zip on zenodo)
