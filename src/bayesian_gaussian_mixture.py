import numpy as np
import pymc as pm


def bayesian_gaussian_mixture_clustering(
    X,
    n_components=3,
    draws=1000,
    tune=1000,
    chains=2,
    random_seed=42,
):
    """
    Fit a Bayesian Gaussian Mixture Model (GMM) to the data X using PyMC.
    This is an unsupervised clustering model: each data point is assigned to a latent cluster,
    and the model infers the cluster means, standard deviations, and mixture weights.

    Args:
        X (np.ndarray): Data matrix, shape (n_samples, n_features)
        n_components (int): Number of mixture components (clusters)
        draws (int): Number of posterior samples per chain (default: 1000)
        tune (int): Number of tuning steps (default: 1000)
        chains (int): Number of MCMC chains (default: 2)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        trace (arviz.InferenceData): PyMC trace object containing posterior samples
        cluster_assignments (np.ndarray): Posterior samples of cluster assignments for each data point, shape (chains, draws, n_samples)
    """
    n_samples, n_features = X.shape
    with pm.Model() as model:
        # Mixture weights
        pi = pm.Dirichlet('pi', a=np.ones(n_components))
        # Cluster means
        mus = pm.Normal('mus', mu=0, sigma=10, shape=(n_components, n_features))
        # Cluster standard deviations
        sigmas = pm.HalfNormal('sigmas', sigma=1, shape=(n_components, n_features))
        # Latent cluster assignment for each data point
        category = pm.Categorical('category', p=pi, shape=n_samples)
        # Likelihood
        obs = pm.Normal('obs', mu=mus[category], sigma=sigmas[category], observed=X)
        trace = pm.sample(draws, tune=tune, chains=chains, random_seed=random_seed, return_inferencedata=True, progressbar=True, cores=1, init="adapt_diag")
    # Extract posterior samples of cluster assignments
    cluster_assignments = trace.posterior['category'].values  # shape: (chains, draws, n_samples)
    return trace, cluster_assignments 