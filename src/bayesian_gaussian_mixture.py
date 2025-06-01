import numpy as np
import pymc as pm
from scipy.stats import norm


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


def posterior_predictive_cluster_assignment(trace, X_new):
    """
    Given a fitted trace and new data, compute the posterior predictive cluster assignment probabilities.

    Args:
        trace (arviz.InferenceData): PyMC trace object containing posterior samples
        X_new (np.ndarray): New data points, shape (n_new, n_features)

    Returns:
        cluster_probs (np.ndarray): Posterior predictive probabilities for each new point and cluster,
                                   shape (n_samples, n_new, n_components)
    """
    # Extract posterior samples
    mus = trace.posterior['mus'].values  # shape: (chains, draws, n_components, n_features)
    sigmas = trace.posterior['sigmas'].values  # shape: (chains, draws, n_components, n_features)
    pis = trace.posterior['pi'].values  # shape: (chains, draws, n_components)
    n_chains, n_draws, n_components, n_features = mus.shape
    n_new = X_new.shape[0]
    n_samples = n_chains * n_draws

    # Reshape for easier iteration
    mus = mus.reshape((n_samples, n_components, n_features))
    sigmas = sigmas.reshape((n_samples, n_components, n_features))
    pis = pis.reshape((n_samples, n_components))

    cluster_probs = np.zeros((n_samples, n_new, n_components))

    for s in range(n_samples):
        for i in range(n_new):
            # Compute likelihood for each cluster
            likelihoods = np.prod(norm.pdf(X_new[i], loc=mus[s], scale=sigmas[s]), axis=1)
            unnorm = pis[s] * likelihoods
            cluster_probs[s, i, :] = unnorm / np.sum(unnorm)
    return cluster_probs  # shape: (n_samples, n_new, n_components) 