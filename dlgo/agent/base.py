from dlgo.goboard import GameState

class Agent:
    def __init__(self) -> None:
        pass

    def select_move(self, game_state: GameState):
        raise NotImplementedError()