
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import inf
from collections import Counter
import itertools
from time import time
GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def alpha_beta_IDS(state, game, depth, eval_fn=None):
    start = time.time()
    player = game.to_move(state)

    def alphabeta(state,alpha, beta, depth):

        def max_value(state, alpha, beta, depth):
            if game.terminal_test(state):
                return game.utility(state, player)
            if time.time() - start > 10 or 0 >= depth: return eval_fn(state)
            v = -np.inf
            for a in game.actions(state):
                v = max(v, min_value(game.result(state, a), alpha, beta, depth - 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth):
            if game.terminal_test(state):
                return game.utility(state, player)
            if time.time() - start > 10 or 0 >= depth: return eval_fn(state)
            v = np.inf
            for a in game.actions(state):
                v = min(v, max_value(game.result(state, a), alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v
        
        return max_value(state, alpha, beta, depth) if state.to_move == 'X' else min_value(state, alpha, beta, depth)

    best_action = None
    for d in range(1, depth):
        if time.time() - start > 10: break
        eval_fn = eval_fn or (lambda state: game.utility(state, player))
        best_score = -np.inf
        beta = np.inf
        for a in game.actions(state):
            v = alphabeta(game.result(state, a), best_score, beta, d)
            if v > best_score:
                best_score = v
                best_action = a
    return best_action

def minmax_decision(state, game, depth, eval_fn = None):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states."""

    player = state.to_move

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth <= 0: return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), depth - 1))
        return v

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth <= 0: return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), depth - 1))
        return v

    # Body of minmax_decision:
    for d in range(1, depth):
        eval_fn = eval_fn or (lambda state: game.utility(state, player))
        v = max(game.actions(state), key=lambda a: min_value(game.result(state, a), d))
    return v

class TicTacToe:

    def __init__(self, h = 3, v = 3, k = 3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0
    
    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k
    
    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

def check_input(user_input, board_size):
    # Strip parentheses from the input
    user_input = user_input.strip("()")
    # Split the input into a and b strings
    a_str, b_str = user_input.split(",")
    if not a_str.isdigit() or not b_str.isdigit():
        return False
    a, b = int(a_str), int(b_str)
    # Check if either a or b is greater than the /board size
    if a > board_size or b > board_size:
        return False
    return True

def human_computer(game, state):
    print("\nStart game!\n")
    game.display(state)
    while not game.terminal_test(state):
        player = state.to_move
        print(f"\n{player}'s turn.")
        if player == 'X':  # User's turn
            move_input = input("Enter your move as (x, y): ")
            if(check_input(move_input, 3)):
                move = eval(move_input)
                state = game.result(state, move)
            else:
                print("Invalid input!\n")
        else:  # Algorithm's turn
            depth = 10  # Set the search depth
            eval_fn = None  # Use the default evaluation function
            move = alpha_beta_IDS(state, game, depth, eval_fn)
            state = game.result(state, move)
            print(f"The algorithm chooses {move}.")
        game.display(state)
    print("\nGame over.")
    if state.utility == 1:
        print("X wins!")
    elif state.utility == -1:
        print("O wins!")
    else:
        print("It's a tie!")

def computer_computer(game, state, algo_type):
    print("\nStart game!\n")
    game.display(state)
    
    results = {'X': 0, 'O': 0, 'Tie': 0}
    total_time = 0
    total_branching = 0

    for _ in range(2):
        state = game.initial  # Reset the game state

        start_time = time.time()
        branching_count = 0
        while not game.terminal_test(state):
            player = state.to_move
            print(f"\n{player}'s turn.")
            if player == 'X':
                depth = 10
                move = alpha_beta_IDS(state, game, depth)
                moves = game.actions(state)
                branching_count += len(moves)  # Count the number of possible moves
            else:
                move = minmax_decision(state, game, depth = 10)
            state = game.result(state, move)
            game.display(state)
        
        print("New game")
        
        game_time = time.time() - start_time
        total_time += game_time
        
        total_branching += branching_count
        
        if state.utility == 1:
            results['X'] += 1
        elif state.utility == -1:
            results['O'] += 1
        else:
            results['Tie'] += 1
    average_branching = total_branching / 10  # Calculate the average branching factor
    
    # Print the results
    print("\nResults:")
    print(f"X Wins: {results['X']}")
    print(f"O Wins: {results['O']}")
    print(f"Ties: {results['Tie']}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average Branching Factor: {average_branching:.2f}")
    return results

def human_human(game, state):
    print("\nStart game!\n")
    game.display(state)
    while not game.terminal_test(state):
        player = state.to_move
        print(f"\n{player}'s turn.")
        move_input = input("Enter your move as (x, y): ")
        if(check_input(move_input, 3)):
            move = eval(move_input)
            state = game.result(state, move)
        else:
            print("Invalid input!\n")
        game.display(state)
    print("\nGame over.")
    if state.utility == 1:
        print("X wins!")
    elif state.utility == -1:
        print("O wins!")
    else:
        print("It's a tie!")

class UltimateTicTacToe:

    TIME_LIMIT = 5

#creating the box for players

def index(x, y):
    x -= 1
    y -= 1
    return ((x//3)*27) + ((x % 3)*3) + ((y//3)*9) + (y % 3)


def box(x, y):
    return index(x, y) // 9


def next_box(i):
    return i % 9


def section_box(b):
    return list(range(b*9, b*9 + 9))

#this should help divide all the boxes nicely 
def print_board(state):
    for row in range(1, 10):
        row_str = ["|"]
        for col in range(1, 10):
            row_str += [state[index(row, col)]]
            if (col) % 3 == 0:
                row_str += ["|"]
        if (row-1) % 3 == 0:
            print("-"*(len(row_str)*2-1))
        print(" ".join(row_str))
    print("-"*(len(row_str)*2-1))


def add_piece(state, move, player):
    if not isinstance(move, int):
        move = index(move[0], move[1])
    return state[: move] + player + state[move+1:]


def update_box_won(state):
    temp_box_win = ["."] * 9
    for b in range(9):
        idxs_box = section_box(b)
        box_str = state[idxs_box[0]: idxs_box[-1]+1]
        temp_box_win[b] = check_box(box_str)
    return temp_box_win

#this should be able to double check boxes during the game
def check_box(box_str):
    global possible_goals
    for idxs in possible_goals:
        (x, y, z) = idxs
        if (box_str[x] == box_str[y] == box_str[z]) and box_str[x] != ".":
            return box_str[x]
    return "."


def possible_moves(last_move):
    global box_won
    if not isinstance(last_move, int):
        last_move = index(last_move[0], last_move[1])
    box_to_play = next_box(last_move)
    idxs = section_box(box_to_play)
    if box_won[box_to_play] != ".":
        pi_2d = [section_box(b) for b in range(9) if box_won[b] == "."]
        possible_indices = list(itertools.chain.from_iterable(pi_2d))
    else:
        possible_indices = idxs
    return possible_indices


def successors(state, player, last_move):
    succ = []
    moves_idx = []
    possible_indexes = possible_moves(last_move)
    for idx in possible_indexes:
        if state[idx] == ".":
            moves_idx.append(idx)
            succ.append(add_piece(state, idx, player))
    return zip(succ, moves_idx)


def print_successors(state, player, last_move):
    for st in successors(state, player, last_move):
        print_board(st[0])


def opponent(p):
    return "O" if p == "X" else "X"


def evaluate_small_box(box_str, player):
    global possible_goals
    score = 0
    three = Counter(player * 3)
    two = Counter(player * 2 + ".")
    one = Counter(player * 1 + "." * 2)
    three_opponent = Counter(opponent(player) * 3)
    two_opponent = Counter(opponent(player) * 2 + ".")
    one_opponent = Counter(opponent(player) * 1 + "." * 2)

    for idxs in possible_goals:
        (x, y, z) = idxs
        current = Counter([box_str[x], box_str[y], box_str[z]])

        if current == three:
            score += 100
        elif current == two:
            score += 10
        elif current == one:
            score += 1
        elif current == three_opponent:
            score -= 100
            return score
        elif current == two_opponent:
            score -= 10
        elif current == one_opponent:
            score -= 1

    return score


def evaluate(state, last_move, player):
    global box_won
    score = 0
    score += evaluate_small_box(box_won, player) * 200
    for b in range(9):
        idxs = section_box(b)
        box_str = state[idxs[0]: idxs[-1]+1]
        score += evaluate_small_box(box_str, player)
    return score





#this will make sure the input is valid as long as the spot is not taken and is still "."
def valid_input(state, move):
    global box_won
    if not (0 < move[0] < 10 and 0 < move[1] < 10):
        return False
    if box_won[box(move[0], move[1])] != ".":
        return False
    if state[index(move[0], move[1])] != ".":
        return False
    return True

#players are able to choose a row and column based on a 1-9 scale
def take_input(state):    
        print("Enter the row and column to place 'X' or 'O' (1-9):")
        x = int(input("Row (1-9): "))
        y = int(input("Column (1-9): "))
        if not valid_input(state, (x,y)):
            raise ValueError
        return (x,y)



#this controls the whole game
def ult_game(state="." * 81):
    global box_won, possible_goals
    possible_goals = [(0, 4, 8), (2, 4, 6)]
    possible_goals += [(i, i + 3, i + 6) for i in range(3)]
    possible_goals += [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(3)]
    box_won = update_box_won(state)
    print_board(state)

    marker_choice = input("Enter 'X' or 'O' to start first: ").upper()
    if marker_choice == "X":
        print ("Player 1 is X")
        print ("Player 2 is O")
        x_turn = True
    else:
        print ("Player 1 is O")
        print ("Player 2 is X")
        x_turn = False  # Flag to check if it's 'X' turn

    while True:
        try:
            user_move = take_input(state)
            if x_turn:
                piece = "X"
            else:
                piece = "O"

            state = add_piece(state, user_move, piece)
            print_board(state)
            box_won = update_box_won(state)
            game_won = check_box(box_won)

            if game_won == "X":
                print("Congratulations Player X Wins!")
                break
            elif game_won == "O":
                print("Congratulations Player O Wins!")
                break
            elif "." not in state:
                print("It's a Tie!")
                break

            # Switch between X and O
            x_turn = not x_turn

            if x_turn == True:
                print("It is X's turn")
            else:
                print ("It is O's turn")

        except ValueError:
            print("Invalid input! Please Try again.")
            print_board(state)
            continue

    return state

if __name__ == "__main__":
    game = TicTacToe()
    state = game.initial

    print("\nLet's play TicTacToe!\n")
    game.display(state)

game_type = input("Please enter 'a' to play against another player, 'b' to play against alpha beta algorithm, or 'c' to watch the algorithm play against itself or 'd' to play ULTIMATE TIC TAC TOE.\n" )

if game_type=="a":
    human_human(game,state)
elif game_type=="b":
    human_computer(game, state)
elif game_type=="c":
    algo_type=input("Please enter 'a' for alpha beta iterative deepening algorithm or 'b' for min max algorithm.\n")
    if algo_type == 'a' or algo_type == 'b':
        results = computer_computer(game, state, algo_type)
         # Plot the results
        labels = ['X Wins', 'O Wins', 'Ties']
        values = [results['X'], results['O'], results['Tie']]
        plt.bar(labels, values)
        plt.xlabel('Results')
        plt.ylabel('Count')
        plt.title('Computer vs. Computer Results')
        plt.show()
    else:
        print("Invalid input")
        sys.exit(0)
    print("Thanks for playing!\n")
elif game_type =='d':
    print("Welcome to Ultimate Tic Tac Toe!\n")
    INITIAL_STATE = "." * 81
    final_state = ult_game(INITIAL_STATE)
    ult_tic_tac = UltimateTicTacToe()   #calls to the ultimate game in the class
