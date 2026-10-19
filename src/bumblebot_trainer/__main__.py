import sys

from .utils import load_config_file


def main(config_path: str):
    config = load_config_file(config_path)

    if config['type'] == 'ssl':
        from .training import SSLTrainer
        trainer = SSLTrainer(config, config_path)
    elif config['type'] in ('pv', 'pv_finetune'):
        from .training import PVTrainer
        trainer = PVTrainer(config, config_path)
    elif config['type'] == 'classification':
        from .training import ClassificationTrainer
        trainer = ClassificationTrainer(config, config_path)
    else:
        raise ValueError(f"Unsupported training type: {config['type']}")

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


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'evaluate':
        args = sys.argv[2:]
        logdir = None
        checkpoint_path = None
        step = 0
        use_mlflow = False
        index = 0
        while index < len(args):
            argument = args[index]
            if argument == '--mlflow':
                use_mlflow = True
                index += 1
                continue

            if argument.startswith('--'):
                if index + 1 >= len(args):
                    print(f'Missing value after {argument}')
                    sys.exit(1)
                value = args[index + 1]
                if argument == '--logdir':
                    if logdir is not None:
                        print('--logdir may only be specified once')
                        sys.exit(1)
                    logdir = value
                elif argument == '--checkpoint':
                    if checkpoint_path is not None:
                        print('--checkpoint may only be specified once')
                        sys.exit(1)
                    checkpoint_path = value
                elif argument == '--step':
                    step = int(value)
                else:
                    print(f'Unknown evaluate option: {argument}')
                    sys.exit(1)
                index += 2
            else:
                print(f'Unexpected evaluate argument: {argument}')
                sys.exit(1)

        if logdir is None:
            print('Usage: python -m bumblebot_trainer evaluate --logdir <run_dir> [--checkpoint <checkpoint.pth>] [--step N] [--mlflow]')
            sys.exit(1)
        evaluation_main(
            logdir,
            checkpoint_path=checkpoint_path,
            use_mlflow=use_mlflow,
            step=step,
        )
    elif len(sys.argv) >= 2 and sys.argv[1] == 'tournament':
        args = sys.argv[2:]
        positional_args = []
        logdirs = []
        book_path = None
        index = 0
        while index < len(args):
            argument = args[index]
            if argument in ('--logdir', '--book'):
                if index + 1 >= len(args):
                    print(f'Missing path after {argument}')
                    sys.exit(1)
                if argument == '--logdir':
                    logdirs.append(args[index + 1])
                else:
                    if book_path is not None:
                        print('--book may only be specified once')
                        sys.exit(1)
                    book_path = args[index + 1]
                index += 2
            elif argument.startswith('--logdir='):
                logdir = argument.partition('=')[2]
                if not logdir:
                    print('Missing path after --logdir=')
                    sys.exit(1)
                logdirs.append(logdir)
                index += 1
            elif argument.startswith('--book='):
                if book_path is not None:
                    print('--book may only be specified once')
                    sys.exit(1)
                book_path = argument.partition('=')[2]
                if not book_path:
                    print('Missing path after --book=')
                    sys.exit(1)
                index += 1
            else:
                positional_args.append(argument)
                index += 1

        if len(positional_args) != 1:
            print('Usage:')
            print('  python -m bumblebot_trainer tournament <tournament_config.yaml> [--logdir <run_dir> ...] [--book <book.csv>]')
            sys.exit(1)
        tournament_main(positional_args[0], logdirs=logdirs, book_path=book_path)
    elif len(sys.argv) >= 2 and sys.argv[1] == 'stockfish-eval':
        args = sys.argv[2:]
        book_path = None
        positional_args = []
        index = 0
        while index < len(args):
            argument = args[index]
            if argument == '--book':
                if index + 1 >= len(args):
                    print('Missing path after --book')
                    sys.exit(1)
                if book_path is not None:
                    print('--book may only be specified once')
                    sys.exit(1)
                book_path = args[index + 1]
                index += 2
            elif argument.startswith('--book='):
                if book_path is not None:
                    print('--book may only be specified once')
                    sys.exit(1)
                book_path = argument.partition('=')[2]
                if not book_path:
                    print('Missing path after --book')
                    sys.exit(1)
                index += 1
            else:
                positional_args.append(argument)
                index += 1

        if len(positional_args) in (3, 4) and positional_args[0] == '--logdir':
            stockfish_eval_main(
                positional_args[2],
                logdir=positional_args[1],
                checkpoint_path=positional_args[3] if len(positional_args) == 4 else None,
                book_path=book_path,
            )
        elif len(positional_args) == 3 and positional_args[0] == '--onnx':
            stockfish_eval_main(
                positional_args[2],
                onnx_checkpoint_path=positional_args[1],
                book_path=book_path,
            )
        elif len(positional_args) == 2 and positional_args[0] == '--consortium':
            stockfish_eval_main(
                positional_args[1],
                book_path=book_path,
            )
        else:
            print('Usage:')
            print('  python -m bumblebot_trainer stockfish-eval --logdir <logdir> <evaluation_config.yaml> [pytorch_checkpoint] [--book <book.csv>]')
            print('  python -m bumblebot_trainer stockfish-eval --onnx <standalone.onnx> <evaluation_config.yaml> [--book <book.csv>]')
            print('  python -m bumblebot_trainer stockfish-eval --consortium <consortium_evaluation_config.yaml> [--book <book.csv>]')
            sys.exit(1)
    elif len(sys.argv) >= 2 and sys.argv[1] == 'export':
        if len(sys.argv) not in (3, 4, 5):
            print("Usage: python -m bumblebot_trainer export <logdir> [output_path] [checkpoint_path]")
            sys.exit(1)
        export_main(
            sys.argv[2],
            output_path=sys.argv[3] if len(sys.argv) >= 4 else None,
            checkpoint_path=sys.argv[4] if len(sys.argv) >= 5 else None,
        )
    else:
        if len(sys.argv) != 2:
            print("Usage: python -m bumblebot_trainer <config_path>")
            print("   or: python -m bumblebot_trainer evaluate --logdir <run_dir> [--checkpoint <checkpoint.pth>] [--step N] [--mlflow]")
            print("   or: python -m bumblebot_trainer tournament <tournament_config.yaml> [--logdir <run_dir> ...] [--book <book.csv>]")
            print("   or: python -m bumblebot_trainer export <logdir> [output_path] [checkpoint_path]")
            print("   or: python -m bumblebot_trainer stockfish-eval --logdir <logdir> <evaluation_config.yaml> [pytorch_checkpoint] [--book <book.csv>]")
            print("   or: python -m bumblebot_trainer stockfish-eval --onnx <standalone.onnx> <evaluation_config.yaml> [--book <book.csv>]")
            print("   or: python -m bumblebot_trainer stockfish-eval --consortium <consortium_evaluation_config.yaml> [--book <book.csv>]")
            sys.exit(1)
        config_path = sys.argv[1]
        main(config_path)
