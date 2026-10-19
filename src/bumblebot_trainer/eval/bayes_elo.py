import numpy as np


def bayes_elo(
        ratings,
        outcomes,
        prior_mean: float,
        prior_std: float,
        elo_min: float,
        elo_max: float,
        elo_step: float,
    ) -> tuple[float, float]:
    """Estimates a model Elo rating and its standard deviation.
    Assumes the opponents have a calibrated elo with no uncertainty. If this is not
    the case, use bayes_elo_ratings.

    This is adapted from https://www.remi-coulom.fr/Bayesian-Elo/.

    Main differences:
    - this implementation uses a elo grid of configurable bounds and step.
    - this implementation uses a gaussian prior over the elo distribution.
    - this implementation assumes the opponents have fixed calibrated elo.
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
        prior_mean: float,
        prior_std: float,
        elo_min: float,
        elo_max: float,
        pinned_ratings: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
    """Estimate player Elo ratings from games between them.

    Can take a pinned_ratings dict as an argument that sets the elo of some players as
    elo anchors.
    """
    names = list(player_names)
    if not names:
        return {}
    if len(names) != len(set(names)):
        raise ValueError('Player names must be unique')
    if prior_std <= 0 or elo_min >= elo_max:
        raise ValueError('Invalid Bayesian Elo prior or rating bounds')

    pinned_ratings = dict(pinned_ratings or {})
    unknown_pinned_players = set(pinned_ratings) - set(names)
    if unknown_pinned_players:
        raise ValueError(
            f'Pinned player(s) are not in the tournament: {sorted(unknown_pinned_players)}'
        )
    for name, rating in pinned_ratings.items():
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            raise ValueError(f'Pinned Elo for {name!r} must be a finite number') from None
        if not np.isfinite(rating) or not elo_min <= rating <= elo_max:
            raise ValueError(
                f'Pinned Elo for {name!r} must be finite and within '
                f'[{elo_min}, {elo_max}], got {rating!r}'
            )
        pinned_ratings[name] = rating

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

    pinned_indices = {
        index[name]: float(rating) for name, rating in pinned_ratings.items()
    }
    free_indices = [i for i in range(len(names)) if i not in pinned_indices]
    free_index = {player_index: i for i, player_index in enumerate(free_indices)}
    ratings = np.full(len(names), float(prior_mean), dtype=np.float64)
    for player_index, rating in pinned_indices.items():
        ratings[player_index] = rating

    scale = np.log(10.0) / 400.0
    prior_precision = 1.0 / float(prior_std) ** 2

    def negative_log_posterior_derivatives():
        gradient = (ratings[free_indices] - prior_mean) * prior_precision
        hessian = np.eye(len(free_indices), dtype=np.float64) * prior_precision
        for white, black, score in valid_games:
            difference = (ratings[white] - ratings[black]) * scale
            probability = 1.0 / (1.0 + np.exp(-np.clip(difference, -40.0, 40.0)))
            residual = (probability - score) * scale
            curvature = probability * (1.0 - probability) * scale ** 2
            white_free = free_index.get(white)
            black_free = free_index.get(black)
            if white_free is not None:
                gradient[white_free] += residual
                hessian[white_free, white_free] += curvature
            if black_free is not None:
                gradient[black_free] -= residual
                hessian[black_free, black_free] += curvature
            if white_free is not None and black_free is not None:
                hessian[white_free, black_free] -= curvature
                hessian[black_free, white_free] -= curvature
        return gradient, hessian

    # Newton optimization of the negative log posterior for unpinned players.
    for _ in range(10):
        if not free_indices:
            break
        gradient, hessian = negative_log_posterior_derivatives()
        step = np.linalg.solve(hessian, gradient)
        updated = np.clip(
            ratings[free_indices] - step,
            elo_min,
            elo_max,
        )
        delta = np.max(np.abs(updated - ratings[free_indices]))
        ratings[free_indices] = updated
        if delta < 1e-5:
            break

    if free_indices:
        _, hessian = negative_log_posterior_derivatives()
        covariance = np.linalg.inv(hessian)
        standard_deviations = np.zeros(len(names), dtype=np.float64)
        standard_deviations[free_indices] = np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        )
    else:
        standard_deviations = np.zeros(len(names), dtype=np.float64)

    return {
        name: (float(ratings[i]), float(standard_deviations[i]))
        for i, name in enumerate(names)
    }
