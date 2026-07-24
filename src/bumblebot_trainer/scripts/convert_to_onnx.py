import os
import sys
import yaml

import torch
import onnx
import chess

from .. import build_model_config
from ..data.utils import encode_board
from ..modeling.ort import PVInferenceModel


def export(logdir):
    config_path = os.path.join(logdir, 'training_config.yaml')
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    model_config = build_model_config(training_config['model'])
    ckpt_path = os.path.join(logdir, 'forecast_model.pth')

    model = PVInferenceModel(model_config)
    missing, unexpected = model.load_state_dict(
      torch.load(ckpt_path, weights_only=True), strict=False
    )
    print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")
    assert not missing, missing[:5]

    model.eval().to('cuda', dtype=torch.float16)

    positions = [
        chess.Board() for _ in range(512)
    ]
    encoded = [encode_board(pos) for pos in positions]
    x = torch.stack([e[0] for e in encoded], dim=0).to('cuda')

    example_input = (x,)

    onnx_program = torch.onnx.export(
        model,
        example_input,
        input_names=['x'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'x': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        dynamo=True,
        verbose=True
    )
    print(onnx_program)

    onnx_program.save(os.path.join(logdir, 'forecast_model.onnx'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_onnx.py <logdir>")
        sys.exit(1)

    export(sys.argv[1])
