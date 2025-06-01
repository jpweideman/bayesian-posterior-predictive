import bayesian_linear_regression_jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

def bayesian_linear_regression_posterior_predictive_jax(
    X_train,
    y_train,
    X_test,
    draws=1000,
    tune=1000,
    chains=1,
    random_seed=42,
):
    """
    Bayesian linear regression with NumPyro (JAX).
    Returns posterior predictive means, stds, samples, and the MCMC trace.
    """
    def model(X, y=None):
        a = numpyro.sample('a', dist.Normal(0, 10))
        b = numpyro.sample('b', dist.Normal(0, 10).expand([X.shape[1]]))
        sigma = numpyro.sample('sigma', dist.HalfNormal(2))
        mu = a + jnp.dot(X, b)
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)

    rng_key = bayesian_linear_regression_jax.random.PRNGKey(random_seed)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=tune, num_samples=draws, num_chains=chains, progress_bar=True)
    mcmc.run(rng_key, X_train, y_train)
    trace = mcmc.get_samples()

    # Posterior predictive for test set
    predictive = Predictive(model, posterior_samples=trace, return_sites=["a", "b", "sigma", "obs"])
    pred_samples = predictive(rng_key, X_test, y=None)
    y_pred_samples = np.array(pred_samples["obs"])  # shape: (draws*chains, n_test)

    y_pred_means = y_pred_samples.mean(axis=0)
    y_pred_stds = y_pred_samples.std(axis=0)

    return y_pred_means, y_pred_stds, y_pred_samples, trace
    
# # Example usage:
# if __name__ == "__main__":
#     # Generate synthetic data
#     np.random.seed(0)
#     X = np.random.randn(100, 2)
#     y = 3 + 2*X[:,0] - X[:,1] + np.random.randn(100)

#     # Split into train/test
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Run Bayesian linear regression with JAX/NumPyro
#     y_pred_means, y_pred_stds, y_pred_samples, trace = bayesian_linear_regression_posterior_predictive_jax(
#         X_train, y_train, X_test, draws=500, tune=500, chains=1, random_seed=42
#     )

#     print("Posterior predictive means (first 5):", y_pred_means[:5])
#     print("Posterior predictive stds (first 5):", y_pred_stds[:5])