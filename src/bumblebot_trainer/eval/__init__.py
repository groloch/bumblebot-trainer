from .bayes_elo import bayes_elo

__all__ = [
    'bayes_elo',
    'PuzzleEvaluator',
    'StockfishEvaluator',
    'EvaluationSuite',
    'run_evaluation_suite',
    'run_evaluation_suite_from_logdir',
]


def __getattr__(name):
    evaluation_exports = (
        'PuzzleEvaluator',
        'StockfishEvaluator',
        'EvaluationSuite',
        'run_evaluation_suite',
        'run_evaluation_suite_from_logdir',
    )
    if name in evaluation_exports:
        from .evaluation import (
            PuzzleEvaluator,
            StockfishEvaluator,
            EvaluationSuite,
            run_evaluation_suite,
            run_evaluation_suite_from_logdir,
        )
        return {
            'PuzzleEvaluator': PuzzleEvaluator,
            'StockfishEvaluator': StockfishEvaluator,
            'EvaluationSuite': EvaluationSuite,
            'run_evaluation_suite': run_evaluation_suite,
            'run_evaluation_suite_from_logdir': run_evaluation_suite_from_logdir,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
