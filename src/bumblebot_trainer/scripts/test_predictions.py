import sys
import yaml

import torch
import chess
from tqdm import tqdm

from .. import build_model_config
from ..modeling.model import ChessModel
from ..data.utils import encode_board
from ..utils import whitewinpercent_to_cp, get_move_from_id
from ..data.game_datasets import SingleGameDataset


def test_predictions(logdir):
    config_path = f'{logdir}/training_config.yaml'
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    model_config = build_model_config(training_config['model'])
    ckpt_path = f'{logdir}/forecast_model.pth'

    model = ChessModel(model_config)
    model.load_state_dict(
        torch.load(ckpt_path, weights_only=True)
    )
    model.eval()
    model.to('cuda', dtype=torch.float16)

    fen = chess.STARTING_FEN

    board = chess.Board(fen)

    x = encode_board(board)
    x = x.unsqueeze(0).to('cuda')

    valid_mask = (x[:, :64] != 0) # True for pieces, False for empty squares
    forecast_depth = model.forecast_head.forecast_depth

    trajectories = torch.zeros(
        (1, 64 * forecast_depth),
        device='cuda',
        dtype=torch.long
    )

    valid_mask_expanded = valid_mask.unsqueeze(1).repeat(1, forecast_depth, 1).view(1, -1)

    with torch.no_grad():
        _, policy_out, value_out = model(
            x
        )

    policy_logits: torch.Tensor
    policy_logits = policy_out.logits.squeeze(0)
    top_k_indices = policy_logits.argsort(descending=True)[:5]
    top_k_probs = policy_logits.softmax(dim=-1)[top_k_indices]

    print("--- Top 5 Moves ---")
    for i, idx in enumerate(top_k_indices):
        m = get_move_from_id(idx.item(), board.turn)
        print(f"{i+1}. {m} (Prob: {top_k_probs[i].item():.4f})")

    win_pct = value_out.logits.squeeze(0).sigmoid().item()
    cp = whitewinpercent_to_cp(win_pct)
    print("\n--- Value Evaluation ---")
    print(f"Win probability (white): {win_pct:.2%}")
    print(f"Centipawns: {cp}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_predictions.py <logdir>")
        sys.exit(1)

    test_predictions(sys.argv[1])
