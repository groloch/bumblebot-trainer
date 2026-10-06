import numpy as np


def bayes_elo(
    ratings,
    outcomes,
    prior_mean: float = 1500.0,
    prior_std: float = 1000.0,
    elo_min: float = 200.0,
    elo_max: float = 3800.0,
    elo_step: float = 2.0,
) -> tuple[float, float]:
    """Estimates a model Elo rating and its standard deviation.

    Args:
        ratings: iterable of puzzle ratings.
        outcomes: iterable of binary outcomes (1 = solved, 0 = not solved),
            aligned with ``ratings``.
        prior_mean: mean of the Gaussian prior over the model Elo.
        prior_std: standard deviation of the Gaussian prior.
        elo_min: lower bound of the evaluation grid.
        elo_max: upper bound of the evaluation grid.
        elo_step: grid resolution.

    Returns:
        tuple[float, float]: posterior mean and standard deviation of the Elo.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if ratings.size == 0:
        return float(prior_mean), float(prior_std)

    assert ratings.shape == outcomes.shape
    prior_std = max(float(prior_std), 1e-6)

    unique_ratings, inverse = np.unique(ratings, return_inverse=True)
    wins = np.bincount(inverse, weights=outcomes, minlength=unique_ratings.size)
    total = np.bincount(inverse, minlength=unique_ratings.size).astype(np.float64)
    losses = total - wins

    grid = np.arange(elo_min, elo_max + elo_step, elo_step, dtype=np.float64)
    scale = np.log(10.0) / 400.0

    log_likelihood = np.zeros_like(grid)
    for rating, won, lost in zip(unique_ratings, wins, losses):
        if won == 0 and lost == 0:
            continue
        d = (rating - grid) * scale
        log_solve = -np.logaddexp(0.0, d)
        log_fail = -np.logaddexp(0.0, -d)
        log_likelihood += won * log_solve + lost * log_fail

    log_prior = -0.5 * ((grid - prior_mean) / prior_std) ** 2
    log_posterior = log_likelihood + log_prior

    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()

    mean = float(np.sum(grid * posterior))
    std = float(np.sqrt(np.sum((grid - mean) ** 2 * posterior)))
    return mean, std
