# tag::alphago_imports[]
import numpy as np
from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
from dlgo import kerasutil
from tqdm import tqdm
import pdb
from itertools import zip_longest
# end::alphago_imports[]


__all__ = [
    'AlphaGoNode',
    'AlphaGoMCTS'
]


# tag::init_alphago_node[]
class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent  # <1>
        self.children = {}  # <1> Move -> AlphaGoNode

        # argmax_a Q(s,a) + u(s,a)에 따라 트리에서 child를 고른다.
        # U(s,a) = P(s,a) / (1 + N(s, a)) 보다 복잡한걸 실제로는 사용.
        # P(s,a) = 강한 정책 신경망으로 구한 prior probability
        # N(s,a) = visit count
        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability  # <2>
        self.u_value = probability  # <3> 
    # <1> Tree nodes have one parent and potentially many children.
    # <2> A node is initialized with a prior probability.
    # <3> The utility function will be updated during search.
    # end::init_alphago_node[]

    # tag::select_node[]
    def select_child(self):
        """argmax_a Q(s,a) + u(s,a)에 따라 트리에서 child를 고른다."""
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + \
                   child[1].u_value)
    # end::select_node[]

    # tag::expand_children[]
    def expand_children(self, moves, probabilities):
        """단말 노드에서는 이 위치에서 가능한 모든 수를 평가하는
        강한 정책 신경망을 호출하고 각각에 AlphaGoNode 인스턴스를 추가한다."""
        # assert len(moves) == len(probabilities) + 2
        for move, prob in zip(moves, probabilities):
        # To handle : ValueError: max() arg is an empty sequence (no point valid moves. only resign & pass)
        # for move, prob in zip_longest(moves, probabilities):
            # if move not in self.children:
            #     if prob is not None:
            self.children[move] = AlphaGoNode(parent=self, probability=prob)
                # else:
                #     self.children[move] = AlphaGoNode(parent=self, probability=1e-9)
        # TODO: fundamental problem. 왜 구현에서 19x19=361가지의 수 밖에 고려하지 않는가?
        # 361 + 2가지의 수를 고려해야 함. 
        self.children[Move.pass_turn()] = AlphaGoNode(parent=self, probability=1/(19*19))
        self.children[Move.resign()] = AlphaGoNode(parent=self, probability=1e-9)
    # end::expand_children[]

    # tag::update_values[]
    def update_values(self, leaf_value):
        """
        u(s,a) = c_u \sqrt{N_p(s,a)} P(s,a) / (1 + N(s,a))
        더 많이 방문된 부모 노드를 가진 노드가 더 많이 활용되도록 한다.

        V(l) = lambda * value(l) + (1 - lambda) * rollout(l)
        즉, 가치 신경망과 정책 롤아웃 (약한 정책 신경망) 을 결합한다.
        Q(s,a) = \sum_{i=1}^{n} V(l_i) / N(s,a)
        """
        if self.parent is not None:
            self.parent.update_values(leaf_value)  # <1>

        self.visit_count += 1  # <2>

        # 이렇게 하면 q_value가 계속 커질 수도 있을텐데 왜지.. 요게 맞지 않나..
        # https://github.com/maxpumperla/deep_learning_and_the_game_of_go/issues/83
        self.q_value = (self.q_value * (self.visit_count-1) + leaf_value) / self.visit_count 
        # self.q_value += leaf_value / self.visit_count  # <3>

        if self.parent is not None:
            c_u = 5 # 고정 상수.
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)  # <4>

    # <1> We update parents first to ensure we traverse the tree top to bottom.
    # <2> Increment the visit count for this node.
    # <3> Add the specified leaf value to the Q-value, normalized by visit count.
    # <4> Update utility with current visit counts.
    # end::update_values[]


# tag::alphago_mcts_init[]
class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100):
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()
    # end::alphago_mcts_init[]

    # tag::alphago_mcts_rollout[]
    def select_move(self, game_state):
        """
        1) 게임 트리 시뮬레이션을 num_simulations회 실행
        2) 각 시뮬레이션에서 지정된 깊이까지 내다보는 `탐색`을 실행
        3) leaf 노드라면 가능한 각 수에 새 AlphaGoNode를 추가하고,
        사전 확률을 구하는 강한 정책 신경망을 사용해서 트리를 `확장`.
        4) non-leaf 노드라면 Q값과 U를 최대로 만드는 수를 선택하여 노드를 `선택`.
        5) 이 시뮬레이션에서 사용한 수를 바둑판에 놓는다.
        6) 지정된 깊이에 도달하면 가치 신경망과 정책 롤아웃이 결합된 
        가치 함수를 구하여 단말 노드를 `평가`
        7) 모든 알파고의 노드를 시뮬레이션에서의 단말값을 사용해서 갱신.
        """
        print("select_move called")
        for simulation in tqdm(range(self.num_simulations)):  # <1>
            current_state = game_state
            node = self.root
            for depth in range(self.depth):  # <2>
                if not node.children:  # <3>
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)  # <4>
                    node.expand_children(moves, probabilities)  # <4>
                
                move, node = node.select_child()  # <5>
                current_state = current_state.apply_move(move)  # <5>

            value = self.value.predict(current_state)  # <6>
            rollout = self.policy_rollout(current_state)  # <6>

            weighted_value = (1 - self.lambda_value) * value + \
                self.lambda_value * rollout  # <7>

            node.update_values(weighted_value)  # <8>
        # <1> From current state play out a number of simulations
        # <2> Play moves until the specified depth is reached.
        # <3> If the current node doesn't have any children...
        # <4> ... expand them with probabilities from the strong policy.
        # <5> If there are children, we can select one and play the corresponding move.
        # <6> Compute output of value network and a rollout by the fast policy.
        # <7> Determine the combined value function.
        # <8> Update values for this node in the backup phase
        # end::alphago_mcts_rollout[]

        # tag::alphago_mcts_selection[]
        move = max(self.root.children, key=lambda move:  # <1>
                   self.root.children.get(move).visit_count)  # <1>

        # self.root = AlphaGoNode() # 이거 하면 밑에 절대 실행 안됨.
        # https://github.com/maxpumperla/deep_learning_and_the_game_of_go/issues/42
        if move in self.root.children:  # <2>
            self.root = self.root.children[move]
            self.root.parent = None
        #     self.root.children = {} # temporary -> 이걸 넣어주면 일단 Illegal move가 안생기긴 하는데, 내부 params는 잘못된 상태.
        else: # 근데 저 이슈대로 하면, else가 절대 실행이 안됨. 그러면 상대가 둔 수가 반영이 안돼서
            # Illegal Move를 두게 되는 오류가 생김. 즉, 무조건 self.root를 manually reset 시켜줘야됨.
            # 그렇게 되면 이전 스텝에서 계산했던 결과를 반영하지 못하게 되어 매번 새로 계산하게 되는 비효율이 생김.
            pdb.set_trace()
        # self.root = AlphaGoNode() # 어쨌든 리셋 필요.
        print("select_move return", move)
        return move
        # <1> Pick most visited child of the root as next move.
        # <2> If the picked move is a child, set new root to this child node.
        # end::alphago_mcts_selection[]

# tag::alphago_policy_probs[]
    def policy_probabilities(self, game_state):
        """
        강한 정책 신경망 예측값을 구하고, 이 예측을 가능한 수의 경우에만 적용하고 정규화한다.
        """
        encoder = self.policy._encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / (np.sum(legal_outputs) + 1e-9 + 1/(19*19))
        return legal_moves, normalized_outputs
# end::alphago_policy_probs[]

# tag::alphago_policy_rollout[]
    def policy_rollout(self, game_state):
        """
        롤아웃 한계에 도달할 때까지 빠른 정책 신경망에 따라 강한 수를 적당히 선택하고 누가 이겼는지 본다.
        다음에 둘 선수가 이겼으면 1, 졌으면 -1, 승자가 결정되지 않았으면 0을 반환한다.
        """
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder
            greedy_move = None
            have_moved = False
            for idx in np.argsort(move_probabilities)[::-1]:
                max_point = encoder.decode_point_index(idx)
                greedy_move = Move(max_point)
                if game_state.is_valid_move(greedy_move):
                    game_state = game_state.apply_move(greedy_move)
                    have_moved = True
                    break
            if not have_moved: # if there were no valid point move
                print("There were no valid point move in policy rollout")
                game_state = game_state.apply_move(Move.pass_turn())

        next_player = game_state.next_player
        winner = game_state.winner()

        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0
# end::alphago_policy_rollout[]


    def serialize(self, h5file):
        """always raises an error"""
        raise IOError("AlphaGoMCTS agent can\'t be serialized" +
                       "consider serializing the three underlying" +
                       "neural networks instead.")

    def reflect_move(self, move: Move) -> None:
        """Not sure if this is right."""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            print('Haven\'t seen this move before:', move)
            self.root = AlphaGoNode()
