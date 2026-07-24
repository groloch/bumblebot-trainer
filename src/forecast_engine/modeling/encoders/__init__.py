from .encoder import Encoder, EncoderOutput, BertEncoder, CNNEncoder, CFEncoder


_encoder_mapping = {
    'bert': BertEncoder,
    'cnn': CNNEncoder,
    'cf': CFEncoder,
}


def build_encoder(name: str, *args, **kwargs) -> Encoder:
    return _encoder_mapping[name](*args, **kwargs)
