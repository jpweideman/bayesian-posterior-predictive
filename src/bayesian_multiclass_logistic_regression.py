import numpy as np
import pymc as pm
from sklearn.preprocessing import OneHotEncoder

def bayesian_multiclass_logistic_regression_posterior_predictive(
    X_train,
    y_train,
    X_test,
    draws=1000,
    tune=1000,
    chains=1,
    random_seed=42,
):
    """
    Fit a Bayesian multinomial (multiclass) logistic regression model using PyMC, sample from the posterior,
    and return posterior predictive probabilities for the test set.

    The model uses a softmax likelihood for multiclass classification, with Normal priors on the coefficients and intercepts.
    Posterior inference is performed using MCMC sampling. For each test point, the function computes the predictive mean,
    standard deviation, and all posterior predictive samples for each class probability.

    Args:
        X_train (np.ndarray): Training features, shape (n_train, n_features)
        y_train (np.ndarray): Training labels, shape (n_train,), integer class labels in [0, n_classes-1]
        X_test (np.ndarray): Test features, shape (n_test, n_features)
        draws (int): Number of posterior samples per chain (default: 1000)
        tune (int): Number of tuning steps (default: 1000)
        chains (int): Number of MCMC chains (default: 1)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        p_pred_means (np.ndarray): Predictive mean probabilities for each test observation and class, shape (n_test, n_classes)
        p_pred_stds (np.ndarray): Predictive stds for each test observation and class, shape (n_test, n_classes)
        p_pred_samples (np.ndarray): Posterior predictive probability samples for each test observation, shape (n_test, n_samples, n_classes)
        trace (arviz.InferenceData): PyMC trace object containing posterior samples
    """
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    y_train_oh = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    eps = 1e-8

    with pm.Model() as model:
        X_data = pm.MutableData("X_data", X_train)
        coefs = pm.Normal('coefs', mu=0, sigma=1, shape=(n_features, n_classes))
        intercept = pm.Normal('intercept', mu=0, sigma=1, shape=(n_classes,))
        logits = pm.math.dot(X_data, coefs) + intercept
        p = pm.Deterministic('p', pm.math.softmax(logits))
        # Custom log-likelihood
        logp = pm.math.log(p + eps)  # Log-probabilities for each class and data point (eps to avoid log(0))
        ll = logp[np.arange(X_train.shape[0]), y_train].sum()  # Select and sum the log-prob of the true class for each data point
        pm.Potential('likelihood', ll)  # Add this total log-likelihood to the model for inference
        trace = pm.sample(
            draws, tune=tune, chains=chains, random_seed=random_seed,
            return_inferencedata=True, progressbar=True, cores=1,
            init="adapt_diag"
        )

    # Posterior predictive for each test point
    coefs_samples = trace.posterior['coefs'].values 
    intercept_samples = trace.posterior['intercept'].values  
    n_chains, n_draws = coefs_samples.shape[:2]
    n_samples = n_chains * n_draws

    coefs_samples = coefs_samples.reshape((n_samples, n_features, n_classes))
    intercept_samples = intercept_samples.reshape((n_samples, n_classes))

    p_pred_means = []
    p_pred_stds = []
    p_pred_samples = []

    for x in X_test:
        logits_pred = np.dot(x, coefs_samples) + intercept_samples 
        p_pred = np.exp(logits_pred)
        p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)  # softmax
        p_pred_means.append(p_pred.mean(axis=0))
        p_pred_stds.append(p_pred.std(axis=0))
        p_pred_samples.append(p_pred)
    return np.array(p_pred_means), np.array(p_pred_stds), np.array(p_pred_samples), trace 