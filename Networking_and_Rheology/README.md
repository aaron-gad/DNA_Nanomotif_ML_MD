# Evaluation of networking behaviour and rheological properties of DNA nanomotif networks

This folder contains scripts and notebooks for simulations of X-shaped nanomotifs that assemble into networks, which can be evaluated with graph-based rheology. Simulation files evaluated in the notebooks were uploaded to zenodo. The analysis is split into the following notebooks:

NCG_thermodyn_x_motif.ipynb:
Thermodynamic insights into X-shaped nanomotifs from oxDNA simulations. MD runs and umbrella sampling are evaluated.

NCG_target_angles_x_motif.ipynb:
Angles within and between arms of X-shaped nanomotifs are derived from oxDNA MD runs. These serve as optimisation targets for coarse-grained bead-spring models.

NCG_BO_opt_validation.ipynb:
To validate Bayesian parameter optimisation for the bead-spring model, targets from random bead-spring parameters are selected and used for iterative optimisation.

NCG_BO_opt_x_motif.ipynb:
Bayesian optimisation of bead-spring parameters, given oxDNA target angle distributions.

NCG_BO_opt_x_motif_m2.ipynb
Bayesian optimisation of bead-spring parameters, using combined optimisation of parameters.

NCG_network_properties.ipynb:
Evaluation of network properties and of rheological behaviour of nanomotif networks assembled with bead-spring simulations.

NCG_ML_rheo_pred.ipynb:
Using Gaussian processes and Bayesian optimisation, connectivity between nanomotifs is predicted given target rheological responses.

NCG_SE_seq_prediction.ipynb:
NUPACK is used in combination with Bayesian optimisation to predict which sticky ends sequences can realise a target connectivity in nanomotif networks.
