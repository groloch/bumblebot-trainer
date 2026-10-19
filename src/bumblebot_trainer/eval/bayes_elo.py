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


def bayes_elo_ratings(
    player_names,
    games,
    prior_mean: float = 1500.0,
    prior_std: float = 1000.0,
    elo_min: float = 200.0,
    elo_max: float = 3800.0,
) -> dict[str, tuple[float, float]]:
    """Estimate relative participant Elo ratings from head-to-head results.

    A logistic Bradley-Terry likelihood is combined with independent Gaussian
    Elo priors. The posterior is approximated by a Laplace approximation at
    its mode, giving a rating mean and standard deviation for every player.
    Draws are represented as half a win, as in the usual Elo score model.

    Each game must contain ``white_model``, ``black_model`` and ``white_score``
    (1 for a White win, 0.5 for a draw, 0 for a Black win).
    """
    names = list(player_names)
    if not names:
        return {}
    if len(names) != len(set(names)):
        raise ValueError('Player names must be unique')
    if prior_std <= 0 or elo_min >= elo_max:
        raise ValueError('Invalid Bayesian Elo prior or rating bounds')

    index = {name: i for i, name in enumerate(names)}
    valid_games = []
    for game in games:
        white = game['white_model']
        black = game['black_model']
        if white not in index or black not in index:
            continue
        score = float(game['white_score'])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f'Invalid game score: {score}')
        valid_games.append((index[white], index[black], score))

    ratings = np.full(len(names), float(prior_mean), dtype=np.float64)
    scale = np.log(10.0) / 400.0
    prior_precision = 1.0 / float(prior_std) ** 2

    # Newton optimization of the negative log posterior.
    for _ in range(100):
        gradient = (ratings - prior_mean) * prior_precision
        hessian = np.eye(len(names), dtype=np.float64) * prior_precision
        for white, black, score in valid_games:
            difference = (ratings[white] - ratings[black]) * scale
            probability = 1.0 / (1.0 + np.exp(-np.clip(difference, -40.0, 40.0)))
            residual = (probability - score) * scale
            curvature = probability * (1.0 - probability) * scale ** 2
            gradient[white] += residual
            gradient[black] -= residual
            hessian[white, white] += curvature
            hessian[black, black] += curvature
            hessian[white, black] -= curvature
            hessian[black, white] -= curvature

        step = np.linalg.solve(hessian, gradient)
        updated = np.clip(ratings - step, elo_min, elo_max)
        if np.max(np.abs(updated - ratings)) < 1e-5:
            ratings = updated
            break
        ratings = updated

    covariance = np.linalg.inv(hessian)
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        name: (float(ratings[i]), float(standard_deviations[i]))
        for i, name in enumerate(names)
    }
