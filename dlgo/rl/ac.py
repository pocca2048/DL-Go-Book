import numpy as np
# import pdb
from keras.optimizer_v1 import SGD

from .experience import ExperienceCollector

from .. import encoders, goboard, kerasutil
from ..agent import Agent
from ..agent.helpers import is_point_an_eye

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


class ACAgent(Agent):
    def __init__(self, model, encoder: encoders.Encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None

        self.last_state_value = 0

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def select_move(self, game_state: goboard.GameState) -> goboard.Move:
        num_moves = self.encoder.board_width * \
            self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        X = np.array([board_tensor])

        actions, values = self.model.predict(X) # Q(s, a)와 V(s)를 예측
        # pdb.set_trace()
        move_probs = actions[0] # actions.shape == (1, game_length)
        estimated_value = values[0][0] # values.shape == (1, 1)

        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(
                game_state.board, point,
                game_state.next_player)
            if move_is_valid and (not fills_own_eye):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value
                    )
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.1, batch_size=128):
        """
        input = 바둑 대국의 상태 (state)
        target = 
          1) policy target = 선택한 수 (action)에 해당하는 위치에 advantage가 들어가는 one-hot vector
            policy gradient 방법에서는 그냥 이긴 경우 모든 수의 확률을 높이고 진 경우 모든 수의 확률을 낮춘다.
            이를 통해 이기는 player가 둘 확률이 가장 높은 action을 softmax를 통해 찾는다.
            **policy = winning player에 대한 appoximator**
            하지만 이러면 학습이 느릴 수 밖에 없다.
            여기에 value 함수를 추가해서 현재 이기고 있다고 판단되고 경기에 이겼다면 그 수는 update될 필요가 낮고,
            이기고 있다고 판단되는데 경기에 졌다면 그 수는 update될 필요가 높다.
            이렇게 advantage라는 가중치를 추가해서 update를 균일하지 않게 해서 학습을 빠르게 하는 것이다.
          2) value target = 이긴 경우 1, 진 경우 -1인 scalar
            tanh에서 나온 값을 mse를 통해 1~-1 사이에서 regression으로 학습.
        """
        opt = SGD(lr=lr)
        self.model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, 0.5])

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))
        # pdb.set_trace()
        for i in range(n):
            action = experience.actions[i] # int
            # pytorch에서 쓰던 cross entropy랑 다르게 `categorical_crossentropy`에서는 
            # sparse하게 index 하나를 넣는게 아니라 dense vector를 넣음 (y_true.shape == y_pred.shape)
            # 심지어 y_true가 합해서 1이 아니어도 됨. 그래서 advantage 같이 1이 아니어도 들어가는 것.
            policy_target[i][action] = experience.advantages[i] # float
            reward = experience.rewards[i] # 1 || -1
            value_target[i] = reward

        self.model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return ACAgent(model, encoder)
