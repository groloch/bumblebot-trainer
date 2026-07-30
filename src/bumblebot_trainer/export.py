import os
import re
from pathlib import Path

import torch
import gguf

from .utils import load_config_file, build_model_config
from .training.utils import model_parameters
from .modeling import ChessModel


_STEP_CKPT_RE = re.compile(r"^(?P<name>.+)_step(?P<step>\d+)\.pth$")


def _find_config_path(logdir: Path) -> Path:
    candidates = sorted(logdir.glob('*.yaml')) + sorted(logdir.glob('*.yml'))
    if not candidates:
        raise FileNotFoundError(f'No config YAML file found in {logdir}')
    if len(candidates) > 1:
        names = ', '.join(p.name for p in candidates)
        raise RuntimeError(f'Ambiguous config files in {logdir}: {names}')
    return candidates[0]


def _resolve_checkpoint_path(logdir: Path, training_name: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        return ckpt

    final_ckpt = logdir / f'{training_name}.pth'
    if final_ckpt.exists():
        return final_ckpt

    best_step = None
    best_path = None
    for candidate in logdir.glob(f'{training_name}_step*.pth'):
        match = _STEP_CKPT_RE.match(candidate.name)
        if match is None:
            continue
        step = int(match.group('step'))
        if best_step is None or step > best_step:
            best_step = step
            best_path = candidate

    if best_path is not None:
        return best_path

    raise FileNotFoundError(
        f'Could not find checkpoint {training_name}.pth or any {training_name}_step*.pth in {logdir}'
    )


def _default_output_path(logdir: Path, training_name: str, output_path: str | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    return logdir / f'{training_name}.gguf'


def _load_chess_model_from_run(logdir: Path, checkpoint_path: str | None):
    config_path = _find_config_path(logdir)
    config = load_config_file(str(config_path))

    run_type = config['type']
    if run_type not in ('pv', 'pvtuner'):
        raise ValueError(
            f"Export currently supports ChessModel checkpoints from 'pv' or 'pvtuner' runs; got type={run_type!r}"
        )

    training_name = config['training']['name']
    ckpt_path = _resolve_checkpoint_path(logdir, training_name, checkpoint_path)

    model_config = build_model_config(config['model'])
    model = ChessModel(model_config)
    params, _ = model_parameters(model)
    print(f'Loaded model of {params/1e6:.2f}M parameters')

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, config_path, ckpt_path


def _prepare_tensor_for_export(name: str, tensor: torch.Tensor):
    tensor = tensor.detach().cpu()

    if tensor.ndim == 2 and name.endswith('.weight'):
        tensor = tensor.transpose(0, 1).contiguous()
    else:
        tensor = tensor.contiguous()

    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)

    return tensor.numpy()


def _add_optional_metadata(writer, config: dict, config_path: Path, ckpt_path: Path):
    if hasattr(writer, 'add_name'):
        writer.add_name('bumblebot')
    if hasattr(writer, 'add_description'):
        writer.add_description('Bumblebot ChessModel exported from bumblebot_trainer')

    if hasattr(writer, 'add_string'):
        writer.add_string('bumblebot.run_type', str(config['type']))
        writer.add_string('bumblebot.config_path', str(config_path))
        writer.add_string('bumblebot.checkpoint_path', str(ckpt_path))
        writer.add_string('bumblebot.encoder_name', str(config['model']['encoder_name']))
        writer.add_string('bumblebot.board_encoding', str(config['data']['encoding']))

    if hasattr(writer, 'add_uint32'):
        writer.add_uint32('bumblebot.hidden_size', int(config['model']['hidden_size']))
        writer.add_uint32('bumblebot.input_size', int(config['model']['input_size']))
        writer.add_uint32('bumblebot.intermediate_size', int(config['model']['intermediate_size']))
        writer.add_uint32('bumblebot.num_layers', int(config['model']['encoder']['num_layers']))
        writer.add_uint32('bumblebot.num_heads', int(config['model']['encoder']['num_heads']))
        if 'compressed_dim' in config['model']['encoder']:
            writer.add_uint32('bumblebot.compressed_dim', int(config['model']['encoder']['compressed_dim']))
        if 'smolgen_dim' in config['model']['encoder']:
            writer.add_uint32('bumblebot.smolgen_dim', int(config['model']['encoder']['smolgen_dim']))
        if 'gen_dim' in config['model']['encoder']:
            writer.add_uint32('bumblebot.gen_dim', int(config['model']['encoder']['gen_dim']))


def export_run_to_gguf(logdir: str, output_path: str | None = None, checkpoint_path: str | None = None):
    logdir_path = Path(logdir)
    if not logdir_path.exists():
        raise FileNotFoundError(f'Logdir not found: {logdir_path}')
    if not logdir_path.is_dir():
        raise NotADirectoryError(f'Not a directory: {logdir_path}')

    model, config, config_path, ckpt_path = _load_chess_model_from_run(logdir_path, checkpoint_path)
    output_path_ = _default_output_path(logdir_path, config['training']['name'], output_path)
    os.makedirs(output_path_.parent, exist_ok=True)

    writer = gguf.GGUFWriter(str(output_path_), 'bumblebot')
    _add_optional_metadata(writer, config, config_path, ckpt_path)

    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        writer.add_tensor(name, _prepare_tensor_for_export(name, tensor))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f'Exported GGUF model to {output_path_}')
    print(f'Config: {config_path}')
    print(f'Checkpoint: {ckpt_path}')
