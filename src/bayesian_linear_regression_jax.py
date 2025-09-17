import jax
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

    rng_key = jax.random.PRNGKey(random_seed)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=tune, num_samples=draws, num_chains=chains, progress_bar=True)
    mcmc.run(rng_key, X_train, y_train)
    trace = mcmc.get_samples()

    # Posterior predictive for test set
    predictive = Predictive(model, posterior_samples=trace, return_sites=["a", "b", "sigma", "obs"])
    pred_samples = predictive(rng_key, X_test, y=None)
    y_pred_samples = np.array(pred_samples["obs"])  

    y_pred_means = y_pred_samples.mean(axis=0)
    y_pred_stds = y_pred_samples.std(axis=0)

    return y_pred_means, y_pred_stds, y_pred_samples, trace
  