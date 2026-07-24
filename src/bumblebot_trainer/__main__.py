import sys
import yaml


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if 'training' in config:
        # yaml doesnt load scientific notation as float
        config['training']['learning_rate'] = float(config['training']['learning_rate'])

        match config['data']['encoding']:
            case 'lc0':
                config['model']['input_size'] = 112
            case 'simplified':
                config['model']['input_size'] = 18

        config['model']['intermediate_size'] = config['model']['encoder']['intermediate_size']

    if config['type'] in ('pv', 'ssl'):
        from .training import build_trainer
        trainer = build_trainer(config, config_path, config['type'])

        trainer.run()
        trainer.wrapup()
        

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m forecast_engine <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
