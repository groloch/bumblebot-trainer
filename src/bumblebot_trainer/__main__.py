import sys

from .utils import load_config_file


def main(config_path: str):
    config = load_config_file(config_path)

    if config['type'] != 'ssl':
        raise ValueError(f"Unsupported training type: {config['type']}")

    from .training import SSLTrainer
    trainer = SSLTrainer(config, config_path)

    trainer.run()
    trainer.wrapup()


def stockfish_eval_main(
    evaluation_config_path: str,
    *,
    logdir: str | None = None,
    checkpoint_path: str | None = None,
    onnx_checkpoint_path: str | None = None,
):
    from .eval.stockfish_evaluation import run_stockfish_evaluation
    return run_stockfish_evaluation(
        evaluation_config_path,
        logdir=logdir,
        checkpoint_path=checkpoint_path,
        onnx_checkpoint_path=onnx_checkpoint_path,
    )


def export_main(logdir: str, output_path: str | None = None, checkpoint_path: str | None = None):
    from .export import export_run_to_gguf
    export_run_to_gguf(logdir, output_path=output_path, checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'stockfish-eval':
        args = sys.argv[2:]
        if len(args) in (3, 4) and args[0] == '--logdir':
            stockfish_eval_main(
                args[2],
                logdir=args[1],
                checkpoint_path=args[3] if len(args) == 4 else None,
            )
        elif len(args) == 3 and args[0] == '--onnx':
            stockfish_eval_main(
                args[2],
                onnx_checkpoint_path=args[1],
            )
        else:
            print('Usage:')
            print('  python -m bumblebot_trainer stockfish-eval --logdir <logdir> <evaluation_config.yaml> [pytorch_checkpoint]')
            print('  python -m bumblebot_trainer stockfish-eval --onnx <standalone.onnx> <evaluation_config.yaml>')
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
            print("   or: python -m bumblebot_trainer export <logdir> [output_path] [checkpoint_path]")
            print("   or: python -m bumblebot_trainer stockfish-eval --logdir <logdir> <evaluation_config.yaml> [pytorch_checkpoint]")
            print("   or: python -m bumblebot_trainer stockfish-eval --onnx <standalone.onnx> <evaluation_config.yaml>")
            sys.exit(1)
        config_path = sys.argv[1]
        main(config_path)
