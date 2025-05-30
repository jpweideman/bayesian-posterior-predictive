import numpy as np
import pymc as pm


def bayesian_linear_regression_posterior_predictive(X_train, y_train, X_test, draws=1000, tune=1000, chains=1, random_seed=42):
    """
    Fit a Bayesian linear regression model using PyMC, sample from the posterior,
    and return posterior predictive samples for the test set.

    Args:
        X_train: np.ndarray, shape (n_train, n_features)
        y_train: np.ndarray, shape (n_train,)
        X_test: np.ndarray, shape (n_test, n_features)
        draws: int, number of posterior samples per chain
        tune: int, number of tuning steps
        chains: int, number of chains for sampling
        random_seed: int, random seed for reproducibility

    Returns:
        y_pred_means: np.ndarray, shape (n_test,)
            Predictive means for each test observation.
        y_pred_stds: np.ndarray, shape (n_test,)
            Predictive stds for each test observation.
        y_pred_samples: np.ndarray, shape (n_samples, n_test)
            Posterior predictive samples for each test observation.
        trace: PyMC trace object (InferenceData)
    """
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sigma=10)
        b = pm.Normal('b', mu=0, sigma=10, shape=X_train.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=2)
        mu = a + pm.math.dot(X_train, b)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
        trace = pm.sample(draws, tune=tune, cores=1, chains=chains, random_seed=random_seed, return_inferencedata=True, progressbar=True)

    # Posterior predictive for each test point
    a_samples = trace.posterior['a'].values.flatten()
    b_samples = trace.posterior['b'].values.reshape(-1, X_train.shape[1])
    sigma_samples = trace.posterior['sigma'].values.flatten()

    y_pred_means = []
    y_pred_stds = []
    y_pred_samples = []

    for x in X_test:
        mu_pred = a_samples + np.dot(b_samples, x)
        y_tilde_pred = np.random.normal(mu_pred, sigma_samples)
        y_pred_means.append(np.mean(y_tilde_pred))
        y_pred_stds.append(np.std(y_tilde_pred))
        y_pred_samples.append(y_tilde_pred)
    return np.array(y_pred_means), np.array(y_pred_stds), np.array(y_pred_samples), trace 