from .encoder import EncoderOutput, Encoder, BertEncoder, CNNEncoder

from .cf import CFEncoder

from .cfc import CFCEncoder


_encoder_mapping = {
    'bert': BertEncoder,
    'cnn': CNNEncoder,
    'cf': CFEncoder,
    'cfc': CFCEncoder
}


def build_encoder(name: str, *args, **kwargs) -> Encoder:
    return _encoder_mapping[name](*args, **kwargs)
