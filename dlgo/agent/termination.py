from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo import scoring


class TerminationStrategy:
    def __init__(self) -> None:
        pass

    def should_pass(self, game_state: goboard.GameState):
        return False

    def should_resign(self, game_state: goboard.GameState):
        return False

class PassWhenOpponentPasses(TerminationStrategy):
    def should_pass(self, game_state: goboard.GameState):
        if game_state.last_move is not None:
            return True if game_state.last_move.is_pass else False

def get(termination):
    if termination == 'opponent_passes':
        return PassWhenOpponentPasses()
    else:
        raise ValueError(f"Unsuppored termination strategy : {termination}")

class TerminationAgent(Agent):
    def __init__(self, agent: Agent, strategy=None) -> None:
        super().__init__()
        self.agent = agent
        self.strategy = strategy if strategy is not None else TerminationStrategy()

    def select_move(self, game_state: goboard.GameState):
        if self.strategy.should_pass(game_state):
            return goboard.Move.pass_turn()
        elif self.strategy.should_resign(game_state):
            return goboard.Move.resign()
        else:
            return self.agent.select_move(game_state)