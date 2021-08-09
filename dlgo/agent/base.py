from dlgo.goboard import GameState, Move

class Agent:
    def __init__(self) -> None:
        pass

    def select_move(self, game_state: GameState) -> Move:
        raise NotImplementedError()