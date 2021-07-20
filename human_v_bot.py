from copy import Error
from dlgo import goboard, gotypes
from dlgo.agent import naive
from dlgo.utils import print_board, print_move, point_from_coords


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = naive.RandomBot()

    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            while True:
                try:
                    human_move = input('-- ')
                    point = point_from_coords(human_move.strip())
                    move = goboard.Move.play(point)
                    print_move(game.next_player, move)
                    game = game.apply_move(move)
                    break
                except Error as e:
                    print('Wrong placement: ', e)
                    print('Try again')
        else:
            move = bot.select_move(game)
            print_move(game.next_player, move)
            game = game.apply_move(move)

if __name__ == "__main__":
    main()
