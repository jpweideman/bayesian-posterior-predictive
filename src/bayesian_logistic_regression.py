import numpy as np
import pymc as pm


def bayesian_logistic_regression_posterior_predictive(
    X_train,
    y_train,
    X_test,
    draws=1000,
    tune=1000,
    chains=1,
    random_seed=42,
):
    """
    Fit a Bayesian logistic regression model using PyMC, sample from the posterior,
    and return posterior predictive probabilities for the test set.

    Args:
        X_train: np.ndarray, shape (n_train, n_features)
        y_train: np.ndarray, shape (n_train,) (binary: 0 or 1)
        X_test: np.ndarray, shape (n_test, n_features)
        draws: int, number of posterior samples per chain
        tune: int, number of tuning steps
        chains: int, number of chains for sampling
        random_seed: int, random seed for reproducibility

    Returns:
        p_pred_means: np.ndarray, shape (n_test,)
            Predictive mean probabilities for each test observation.
        p_pred_stds: np.ndarray, shape (n_test,)
            Predictive stds for each test observation.
        p_pred_samples: np.ndarray, shape (n_test, n_samples)
            Posterior predictive probability samples for each test observation.
        trace: PyMC trace object (InferenceData)
    """
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sigma=10)
        b = pm.Normal('b', mu=0, sigma=10, shape=X_train.shape[1])
        logits = a + pm.math.dot(X_train, b)
        p = pm.Deterministic('p', pm.math.sigmoid(logits))
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
        trace = pm.sample(draws, tune=tune, chains=chains, random_seed=random_seed, return_inferencedata=True, progressbar=True, cores=1, target_accept=0.95)

    a_samples = trace.posterior['a'].values.flatten()
    b_samples = trace.posterior['b'].values.reshape(-1, X_train.shape[1])

    p_pred_means = []
    p_pred_stds = []
    p_pred_samples = []

    for x in X_test:
        logits_pred = a_samples + np.dot(b_samples, x)
        p_pred = 1 / (1 + np.exp(-logits_pred))
        p_pred_means.append(np.mean(p_pred))
        p_pred_stds.append(np.std(p_pred))
        p_pred_samples.append(p_pred)
    return np.array(p_pred_means), np.array(p_pred_stds), np.array(p_pred_samples), trace 