from __future__ import \
    annotations  # for self type annotation. https://stackoverflow.com/q/33533148

import pdb

import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard_fast import GameState, Move
from keras.optimizer_v1 import SGD

from .encoder import ZeroEncoder
from .experience import ZeroExperienceCollector

__all__ = [
    'ZeroAgent',
]


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
    
    def __repr__(self) -> str:
        return f"prior: {self.prior}, visit_count: {self.visit_count}, total_value: {self.total_value}"


class ZeroTreeNode:
    def __init__(self, state: GameState, value: float, priors: dict, 
            parent: ZeroTreeNode, last_move: Move):
        # assert value is not None and priors is not None
        self.state = state
        self.value = value
        self.parent = parent                      # <1>
        self.last_move = last_move                # <1>
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        # assert len(self.branches) > 0
        self.children = {}                        # <2>

    def moves(self):                              # <3>
        return self.branches.keys()               # <3>

    def add_child(self, move, child_node):        # <4>
        self.children[move] = child_node          # <4>

    def has_child(self, move):                    # <5>
        return move in self.children              # <5>

    def get_child(self, move):
        return self.children.get(move)

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        if move not in self.branches:
            raise ValueError("move not in self.branches")
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        if move not in self.branches:
            raise ValueError("move not in self.branches")
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


class ZeroAgent(Agent):
    def __init__(self, model, encoder: ZeroEncoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

    def select_move(self, game_state: GameState):
        root = self.create_node(game_state)           # <1>

        for i in range(self.num_rounds):              # <2>
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):          # <3>
                node = node.get_child(next_move)
                next_move = self.select_branch(node)


            new_state = node.state.apply_move(next_move)

            # Added to avoid ValueError: max() arg is an empty sequence
            if new_state.is_over():
                break

            child_node = self.create_node(
                new_state, parent=node, move=next_move)

            move = next_move
            value = -1 * child_node.value             # <1>
            while node is not None: # should be child_node?
                node.record_visit(move, value)
                move = node.last_move # why??
                node = node.parent
                value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
            ])
            if not np.any(visit_counts):
                print("No visit counts!")
                # pdb.set_trace()
            else:
                self.collector.record_decision(
                    root_state_tensor, visit_counts)
        else:
            raise Exception("No collector!")

        return max(root.moves(), key=root.visit_count) # argmax

    def set_collector(self, collector: ZeroExperienceCollector):
        self.collector = collector

    def select_branch(self, node: ZeroTreeNode) -> Move:
        """
        Q + cP*\sqrt{N}/(1+N)
        """
        total_n = node.total_visit_count

        def score_branch(move: Move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)             # <1>

    def create_node(self, game_state: GameState, move: Move = None, parent: ZeroTreeNode = None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])                 # <1>
        priors, values = self.model.predict(model_input)
        priors = priors[0]                                     # <2>
        # Add Dirichlet noise to the root node.
        # Exploration을 높이는 방법 중 하나이다.
        if parent is None:
            noise = np.random.dirichlet( # 0.03 (concentration param) 대신 클 수를 쓸수록 고루 분포되고
                0.03 * np.ones_like(priors)) # 0.03 대신 작은 수를 쓸수록 하나에 집중되는 형태.
            priors = 0.75 * priors + 0.25 * noise
        value = values[0][0]                                   # <2>
        move_priors = {                                        # <3>
            self.encoder.decode_move_index(idx): p             # <3>
            for idx, p in enumerate(priors)                    # <3>
        }                                                      # <3>
        try:
            new_node = ZeroTreeNode(state=game_state, value=value,
                priors=move_priors, parent=parent, last_move=move)
        except:
            pdb.set_trace()
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):     # <1>
        """
        alphazero에서는 정책 출력과 가치 출력이 있다.
        정책 출력은 다른 정책 신경망처럼 가능한 수에서 확률 분포를 만들어낸다.
        다만, 차이점은 이기는 agent가 둘 수를 예측하는 게 아니라, agent가 트리 탐색 중 방문한 횟수 (비율)를 학습한다.
        결국 트리 탐색하는 공식에 따르면 많이 선택하는 가지 = 높은 값의 가지이므로 그렇다.
        따라서 모델 안에 트리를 내장하고 있다고 볼 수 있는 것이다.
        """
        num_examples = experience.states.shape[0]

        model_input = experience.states
        print(experience.visit_counts, experience.visit_counts.shape)
        visit_sums = np.sum(                                    # <2>
            experience.visit_counts, axis=1).reshape(           # <2>
            (num_examples, 1))                                  # <2>
        action_target = experience.visit_counts / visit_sums    # <2>

        value_target = experience.rewards

        self.model.compile(
            SGD(lr=learning_rate),
            loss=['categorical_crossentropy', 'mse'])
        self.model.fit(
            model_input, [action_target, value_target],
            batch_size=batch_size)
