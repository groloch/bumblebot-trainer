from .trainer import Trainer

from .trainers import PVTrainer
from .ssl import SSLTrainer, LegalAttacksTrainer


def build_trainer(config, config_path, type):
    if type == 'pv':
        return PVTrainer(config, config_path)
    elif type == 'ssl':
        return SSLTrainer(config, config_path)
    elif type == 'legal_attacks':
        return LegalAttacksTrainer(config, config_path)
    raise ValueError(f'Unknown trainer type: {type}')
