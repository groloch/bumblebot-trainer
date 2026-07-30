import sys

from .utils import load_config_file


def main(config_path: str):
    config = load_config_file(config_path)

    if config['type'] in ('pv', 'ssl', 'legal_attacks', 'pvtuner'):
        from .training import build_trainer
        trainer = build_trainer(config, config_path, config['type'])

        trainer.run()
        trainer.wrapup()


def export_main(logdir: str, output_path: str | None = None, checkpoint_path: str | None = None):
    from .export import export_run_to_gguf
    export_run_to_gguf(logdir, output_path=output_path, checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'export':
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
            sys.exit(1)
        config_path = sys.argv[1]
        main(config_path)
