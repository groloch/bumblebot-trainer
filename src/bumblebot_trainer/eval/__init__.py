from .bayes_elo import bayes_elo

__all__ = ['bayes_elo', 'PuzzleEvaluator']


def __getattr__(name):
    if name == 'PuzzleEvaluator':
        from .evaluation import PuzzleEvaluator
        return PuzzleEvaluator
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
