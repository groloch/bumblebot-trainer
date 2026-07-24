from .trainer import Trainer

def build_trainer(config, config_path, type):
    if type == 'pv':
        from .trainers import PVTrainer
        return PVTrainer(config, config_path)
    elif type == 'ssl':
        from .trainers import SSLTrainer
        return SSLTrainer(config, config_path)
    raise ValueError(f'Unknown trainer type: {type}')
