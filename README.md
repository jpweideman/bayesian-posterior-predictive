## Bayesian Posterior Predictive Distribution Simulation

This repo contains Bayesian modeling utilities and notebooks for posterior predictive simulation. The focus is on making posterior predictive inferences for unseen data.

### Contents
- `src/bayesian_gaussian_mixture.py`: PyMC Gaussian Mixture Model (GMM) with:
  - `bayesian_gaussian_mixture_clustering(X, ...)` – fits a Bayesian GMM
  - `posterior_predictive_cluster_assignment(trace, X_new)` – posterior predictive cluster probs for new points 
- `src/bayesian_linear_regression.py`: Bayesian linear regression (PyMC)
- `src/bayesian_linear_regression_jax.py`: Bayesian linear regression (NumPyro/JAX)
- `src/bayesian_logistic_regression.py`: Bayesian logistic regression
- `src/bayesian_multiclass_logistic_regression.py`: Bayesian multiclass logistic regression 
- `notebooks/unsupervised_learning.ipynb`: Clustering with Bayesian GMM, PCA plots, posterior predictive on the test set, label-alignment for averaging
- `notebooks/supervised_learning.ipynb`: Regression/classification demos
- `notebooks/posterior_sampling.ipynb`: Posterior sampling utilities
