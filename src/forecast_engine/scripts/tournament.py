import os
import sys
import yaml

import torch
from tqdm import tqdm
import chess

from ..modeling.model import ChessModel
from ..utils import encode_board, get_move_id, get_move_from_id, ChessConstants
from .. import build_model_config
from ..config.modeling_configs import ModelConfig


class Player:
    def __init__(self, name, model: ChessModel):
        self.name = name
        self.model: ChessModel = model

        self.elo = 1500
        self.k = 10

        self.model.to(torch.float16)
        self.model.eval()

        self.score = 0

    def ready_up(self):
        self.model.to('cuda', dtype=torch.float16)

    def get_move(self, board):
        x, hm, epsq = encode_board(board)
        x = x.unsqueeze(0).to('cuda')
        hm = hm.unsqueeze(0).to('cuda', dtype=torch.float16)
        epsq = epsq.unsqueeze(0).to('cuda')

        legal_mask = torch.full(
            (ChessConstants.NUM_POLICY_CLASSES,),
            float('-inf'),
            device='cuda',
            dtype=torch.float16
        )
        for move in board.legal_moves:
            move_id = get_move_id(move, board.turn)
            legal_mask[move_id] = 0.0


        with torch.inference_mode():
            _, policy_out, _, _ = self.model(x, hm, epsq)
            policy_out = policy_out.logits.squeeze(0)
            policy_out = policy_out + legal_mask
            probs = torch.softmax(policy_out, dim=0)
            move_id = torch.multinomial(probs, 1).cpu().item()

        move = get_move_from_id(move_id, board.turn)
        return move

    def register_result(self, opponent_elo, result, color):
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))
        if result == 1 and color == chess.WHITE:
            score = 1
        elif result == 0 and color == chess.BLACK:
            score = 1
        elif result == 0.5:
            score = 0.5
        else:
            score = 0

        self.elo += self.k * (score - expected_score)
        self.score += score

        self.model.to('cpu')


def get_matching_dirs(prefix):
    subdirs = os.listdir('logs')
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join('logs', d))]
    subdirs = [d for d in subdirs if d.startswith(prefix)]
    return subdirs

def play_game(player1: Player, player2: Player):
    board = chess.Board()

    player1.ready_up()
    player2.ready_up()

    while not board.is_game_over(claim_draw=True):
        move1 = player1.get_move(board)
        board.push(move1)
        if board.is_game_over(claim_draw=True):
            break
        move2 = player2.get_move(board)
        board.push(move2)

    result = board.result(claim_draw=True)
    if result == '1-0':
        return 1
    elif result == '0-1':
        return 0
    else:
        return 0.5

def run_tournament(prefix, rounds=10):
    players: list[Player] = []

    subdirs = get_matching_dirs(prefix)
    for subdir in subdirs:
        dirpath = os.path.join('logs', subdir)
        config_path = os.path.join(dirpath, 'training_config.yaml')
        model_path = os.path.join(dirpath, 'forecast_model.pth')

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = build_model_config(config_dict['model'])

        model = ChessModel(config)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        player = Player(subdir[len(prefix):], model)
        players.append(player)

    n_players = len(players)
    print(f'Found {n_players} players: {[p.name for p in players]}')
    for round in tqdm(range(rounds), desc='Playing tournament'):
        for i in range(n_players):
            for j in range(n_players):
                if i == j:
                    continue
                player1 = players[i]
                player2 = players[j]

                result = play_game(player1, player2)

                player1.register_result(player2.elo, result, chess.WHITE)
                player2.register_result(player1.elo, result, chess.BLACK)

    players.sort(key=lambda p: p.score, reverse=True)

    for player in players:
        print(f"{player.name}: Score={player.score}, Elo={player.elo:.2f}")



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python -m forecast_engine.scripts.tournament <model_logdir_prefix> <n_rounds>")
        sys.exit(1)
    run_tournament(sys.argv[1], int(sys.argv[2]))