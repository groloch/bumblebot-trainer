import sys
import yaml

import torch
import chess
from tqdm import tqdm

from .. import build_model_config
from ..modeling.model import ChessModel
from ..utils import encode_board, whitewinpercent_to_cp, get_move_from_id, ForecastVocabulary
from ..data.game_datasets import SingleGameDataset


def get_trajectory(model, x, hm, epsq, valid_mask_expanded, num_steps=10):
    forecast_depth = model.forecast_head.forecast_depth
    
    trajectories = torch.zeros(
        (1, 64 * forecast_depth),
        device=x.device,
        dtype=torch.long
    )
    trajectories[valid_mask_expanded] = ForecastVocabulary.MASK_TOKEN_ID(forecast_depth)

    mask = valid_mask_expanded.clone()
    num_to_predict = mask.sum().item()
    
    steps_schedule = [num_to_predict // num_steps] * num_steps
    for i in range(num_to_predict % num_steps):
        steps_schedule[i] += 1
        
    for step, to_predict in tqdm(enumerate(steps_schedule)):
        if to_predict == 0:
            continue
            
        with torch.no_grad():
            _, _, _, forecast_out = model(
                x, hm, epsq, trajectories=trajectories, trajectories_padding_mask=valid_mask_expanded
            )
            
        forecast_logits = forecast_out.logits.squeeze(0)
        horizon_logits = forecast_out.horizon_logits.squeeze(0)
        
        probs = forecast_logits.softmax(dim=-1)
        max_probs, preds = probs.max(dim=-1)
        h_preds = horizon_logits.argmax(dim=-1)
        
        max_probs[~mask.squeeze(0)] = -1.0
        
        _, topk_indices = max_probs.topk(to_predict)
        
        combined_token = preds[topk_indices] + h_preds[topk_indices] * ForecastVocabulary.HORIZON_OFFSET
        trajectories.squeeze(0)[topk_indices] = combined_token
        mask.squeeze(0)[topk_indices] = False
        
    final_preds = trajectories.squeeze(0)
    forecast_preds = final_preds % ForecastVocabulary.HORIZON_OFFSET
    horizon_preds = final_preds // ForecastVocabulary.HORIZON_OFFSET
    
    return forecast_preds, horizon_preds

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

    x, hm, epsq = encode_board(board)
    x = x.unsqueeze(0).to('cuda')
    hm = hm.unsqueeze(0).to('cuda', dtype=torch.float16)
    epsq = epsq.unsqueeze(0).to('cuda')

    valid_mask = (x[:, :64] != 0) # True for pieces, False for empty squares
    forecast_depth = model.forecast_head.forecast_depth

    trajectories = torch.zeros(
        (1, 64 * forecast_depth),
        device='cuda',
        dtype=torch.long
    )

    valid_mask_expanded = valid_mask.unsqueeze(1).repeat(1, forecast_depth, 1).view(1, -1)
    trajectories[valid_mask_expanded] = ForecastVocabulary.MASK_TOKEN_ID(forecast_depth)

    with torch.no_grad():
        _, policy_out, value_out, forecast_out = model(
            x, hm, epsq, trajectories=trajectories, trajectories_padding_mask=valid_mask_expanded
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

    print("\n--- Piece Trajectories ---")
    forecast_preds, horizon_preds = get_trajectory(
        model, x, hm, epsq, valid_mask_expanded, num_steps=100
    )

    horizon_names = {0: "short", 1: "medium", 2: "long"}
    valid_sq = valid_mask[0].nonzero().squeeze()

    for sq_tensor in valid_sq:
        sq = sq_tensor.item()
        piece = board.piece_at(sq)
        if piece is None:
            continue

        piece_name = piece.symbol()
        sq_name = chess.square_name(sq)

        traj_str = []
        for step in range(forecast_depth):
            idx = step * 64 + sq
            f_pred = forecast_preds[idx].item()
            h_pred = horizon_preds[idx].item()

            if f_pred < 64:
                target = chess.square_name(f_pred)
            elif f_pred == 64:
                target = "TAKEN"
            else:
                target = f"PROMOTE_{f_pred-64}"

            traj_str.append(f"{target} ({horizon_names.get(h_pred, 'unk')})")

            if f_pred == 64:
                break

        print(f"[{sq_name}] {piece_name}: {' -> '.join(traj_str)}")

def test_forecast_data():
    pgn = 'test.pgn'
    dataset = SingleGameDataset(
        pgn=pgn,
        forecast_depth=5,
    )

    _, _, _, td = dataset[0]
    forecast = td['forecast']
    horizon = td['horizon']
    mask = td['forecast_mask']

    print(forecast.shape)
    print(horizon.shape)
    print(mask.shape)

    board: chess.Board = dataset.board
    for s in range(64):
        piece = board.piece_at(s)
        if piece is not None:
                trajectory = forecast[:, s].numpy()
                horizons = horizon[:, s].numpy()
                masks = mask[:, s].numpy()

                text = f'{chess.square_name(s)} ({piece.symbol()}): '
                for f, h, m in zip(trajectory, horizons, masks):
                    if not m:
                        continue
                    if f == 64:
                        text += f"TAKEN ({h}) -> "
                    elif f < 64:
                        text += f"{chess.square_name(f)} ({h}) -> "
                    else:
                        text += f"PROMOTE_{f-64} ({h}) -> "
                print(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_predictions.py <logdir>")
        sys.exit(1)

    if sys.argv[1] == "data":
        test_forecast_data()
    else:
        test_predictions(sys.argv[1])
