import numpy as np
from dlgo import encoders, kerasutil
from dlgo.agent import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo.encoders import Encoder
from dlgo.goboard_fast import GameState, Move
from dlgo.rl.experience import ExperienceBuffer, ExperienceCollector
from keras.optimizer_v1 import SGD


class QAgent(Agent):
    def __init__(self, model, encoder: Encoder) -> None:
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def select_move(self, game_state: GameState) -> Move:
        board_tensor = self.encoder.encode(game_state)

        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor)
        
        if not moves:
            return Move.pass_turn()
        
        num_moves = len(moves)
        board_tensors = np.array(board_tensors)
        move_vectors = np.zeros((num_moves, self.encoder.num_points()))

        for i, move in enumerate(moves):
            move_vectors[i][move] = 1 # one-hot encoding
        
        values = self.model.predict([board_tensors, move_vectors]) # Q(s, a) = value
        values = values.reshape(len(moves)) # N x 1

        ranked_moves = self.rank_moves_eps_greedy(values)

        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(state=board_tensor, action=moves[move_idx])
                return Move.play(point=point)
        
        return Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        ranked_moves = np.argsort(values)
        return ranked_moves[::-1]

    def train(self, experience: ExperienceBuffer, lr=0.1, batch_size=128) -> None:
        opt = SGD(lr=lr)
        self.model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            actions[i][action] = 1
            y[i] = reward

        self.model.fit(
            [experience.states, actions],
            y,
            batch_size=batch_size,
            epochs=1,
        )

def load_q_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return QAgent(model, encoder)
