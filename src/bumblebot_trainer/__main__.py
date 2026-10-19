import argparse
import sys

from .utils import load_config_file


COMMANDS = {'evaluate', 'tournament', 'stockfish-eval', 'export'}


def main(config_path: str):
    config = load_config_file(config_path)
    training_type = config['type']
    from .training import Trainer, SSLTrainer, PVTrainer, ClassificationTrainer
    trainers: dict[str, type[Trainer]] = {
        'ssl': SSLTrainer,
        'pv': PVTrainer,
        'pv_finetune': PVTrainer,
        'classification': ClassificationTrainer,
    }
    if training_type not in trainers:
        raise ValueError(f'Unsupported training type: {training_type}')

    trainer = trainers[training_type](config, config_path)
    trainer.run()
    trainer.wrapup()


def evaluation_main(
    logdir: str,
    *,
    checkpoint_path: str | None = None,
    use_mlflow: bool = False,
    step: int = 0,
):
    from .eval import run_evaluation_suite_from_logdir

    return run_evaluation_suite_from_logdir(
        logdir,
        checkpoint_path=checkpoint_path,
        use_mlflow=use_mlflow,
        step=step,
    )


def stockfish_eval_main(
    evaluation_config_path: str,
    *,
    logdir: str | None = None,
    checkpoint_path: str | None = None,
    onnx_checkpoint_path: str | None = None,
    book_path: str | None = None,
):
    from .eval.stockfish_evaluation import run_stockfish_evaluation

    return run_stockfish_evaluation(
        evaluation_config_path,
        logdir=logdir,
        checkpoint_path=checkpoint_path,
        onnx_checkpoint_path=onnx_checkpoint_path,
        book_path=book_path,
    )


def export_main(logdir: str, output_path: str | None = None, checkpoint_path: str | None = None):
    from .export import export_run_to_gguf

    export_run_to_gguf(logdir, output_path=output_path, checkpoint_path=checkpoint_path)


def tournament_main(
    tournament_config_path: str,
    *,
    logdirs: list[str] | None = None,
    book_path: str | None = None,
):
    from .eval.tournament_evaluation import run_tournament

    return run_tournament(tournament_config_path, logdirs=logdirs, book_path=book_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m bumblebot_trainer',
        description='Train Bumblebot models or run model evaluation tools.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    evaluate = commands.add_parser(
        'evaluate', help='Evaluate a run on puzzle and Stockfish positions'
    )
    evaluate.add_argument('--logdir', required=True, help='Training run directory')
    evaluate.add_argument(
        '--checkpoint', help='Checkpoint path (defaults to the latest checkpoint)'
    )
    evaluate.add_argument('--step', type=int, default=0, help='MLflow metric step')
    evaluate.add_argument('--mlflow', action='store_true', help='Log metrics to MLflow')

    tournament = commands.add_parser('tournament', help='Run an all-against-all tournament')
    tournament.add_argument('config', help='Tournament YAML config')
    tournament.add_argument(
        '--logdir', action='append', default=[], help='Additional participant run directory'
    )
    tournament.add_argument('--book', help='Opening book CSV')

    stockfish = commands.add_parser(
        'stockfish-eval', help='Play paired games against Stockfish'
    )
    stockfish.add_argument(
        'evaluation_config', nargs='?', help='Stockfish evaluation YAML config'
    )
    source = stockfish.add_mutually_exclusive_group()
    source.add_argument('--logdir', help='Training run directory')
    source.add_argument('--onnx', help='Standalone ONNX policy checkpoint')
    source.add_argument(
        '--consortium', metavar='CONFIG', help='Evaluation config containing a consortium'
    )
    stockfish.add_argument(
        'checkpoint', nargs='?', help='Optional PyTorch checkpoint (with --logdir)'
    )
    stockfish.add_argument(
        '--checkpoint', dest='checkpoint_option', help='PyTorch checkpoint override'
    )
    stockfish.add_argument('--book', help='Opening book CSV')

    export = commands.add_parser('export', help='Export a training run to GGUF')
    export.add_argument('logdir', help='Training run directory')
    export.add_argument('output_path', nargs='?', help='Output GGUF path')
    export.add_argument('checkpoint', nargs='?', help='Checkpoint path')
    return parser


def _run_command(args: argparse.Namespace) -> None:
    if args.command == 'evaluate':
        evaluation_main(
            args.logdir,
            checkpoint_path=args.checkpoint,
            use_mlflow=args.mlflow,
            step=args.step,
        )
    elif args.command == 'tournament':
        tournament_main(args.config, logdirs=args.logdir, book_path=args.book)
    elif args.command == 'stockfish-eval':
        config_path = args.consortium or args.evaluation_config
        if config_path is None:
            raise SystemExit('stockfish-eval requires an evaluation config')
        if args.consortium and args.evaluation_config:
            raise SystemExit(
                'Pass the consortium config positionally or with --consortium, not both'
            )
        if args.checkpoint and args.checkpoint_option:
            raise SystemExit(
                'Specify the checkpoint positionally or with --checkpoint, not both'
            )
        stockfish_eval_main(
            config_path,
            logdir=args.logdir,
            onnx_checkpoint_path=args.onnx,
            checkpoint_path=args.checkpoint_option or args.checkpoint,
            book_path=args.book,
        )
    else:
        export_main(args.logdir, args.output_path, args.checkpoint)


def cli(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in COMMANDS and argv[0] not in ('-h', '--help'):
        training_parser = argparse.ArgumentParser(
            prog='python -m bumblebot_trainer',
            description='Train a model using the given YAML config.',
        )
        training_parser.add_argument('config', help='Training YAML config')
        main(training_parser.parse_args(argv).config)
        return

    _run_command(build_parser().parse_args(argv))


if __name__ == '__main__':
    cli()
