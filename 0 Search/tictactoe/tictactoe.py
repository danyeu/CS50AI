"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


# Returns starting state of the board.
def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


# Returns player who has the next turn on a board.
def player(board):
    count = 0
    for row in board:
        for col in row:
            if not col == EMPTY:
                count += 1
    if count % 2 == 0:
        return X
    return O


# Returns set of all possible actions (i, j) available on the board.
def actions(board):
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


# Returns the board that results from making move (i, j) on the board.
def result(board, action):
    if not board[action[0]][action[1]] == EMPTY:
        # if action is a taken cell, raise error
        raise Exception("bad action")
    else:
        resulting_board = copy.deepcopy(board)
        # fill action cell with whose turn it is
        resulting_board[action[0]][action[1]] = player(board)
        return resulting_board


# Returns the winner of the game, if there is one.
def winner(board):
    # X worth 1, O worth -1
    # winner exists if the sum of any row, col, or diag is +3 or -3
    sum_row = [0, 0, 0]
    sum_col = [0, 0, 0]
    sum_diag = [0, 0]

    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                sum_row[i] += 1
                sum_col[j] += 1
                if i == j:
                    sum_diag[0] += 1
                if i == 2 - j:
                    sum_diag[1] += 1
            elif board[i][j] == O:
                sum_row[i] -= 1
                sum_col[j] -= 1
                if i == j:
                    sum_diag[0] -= 1
                if i == 2 - j:
                    sum_diag[1] -= 1

    for _ in [sum_row, sum_col, sum_diag]:
        for item in _:
            if item == 3:
                return X
            elif item == -3:
                return O

    # if no winner return None
    return None


# Returns True if game is over, False otherwise.
def terminal(board):
    if winner(board) is not None:
        return True

    for row in board:
        if EMPTY in row:
            return False

    return True


# Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
def utility(board):
    winning_player = winner(board)
    if winning_player == X:
        return 1
    elif winning_player == O:
        return -1
    else:
        return 0


# Returns the optimal action for the current player on the board.
def minimax(board):
    optimal_action = None

    if player(board) == X:
        best_utility = -2
        for action in actions(board):
            new_utility = minvalue(result(board, action))
            if new_utility > best_utility:
                best_utility = new_utility
                optimal_action = action
    elif player(board) == O:
        best_utility = 2
        for action in actions(board):
            new_utility = maxvalue(result(board, action))
            if new_utility < best_utility:
                best_utility = new_utility
                optimal_action = action
    return optimal_action


# Returns max utility possible from given board assuming optimal play.
def maxvalue(board):
    # immediately return utility for terminal boards
    if terminal(board):
        return utility(board)
    # initialise max value at below all possible outcomes
    v = -2
    # for all remaining actions, find the max utility given the opponent minimises their utility in the next move
    for action in actions(board):
        v = max(v, minvalue(result(board, action)))
    return v


# Returns min utility possible from given board assuming optimal play.
def minvalue(board):
    # immediately return utility for terminal boards
    if terminal(board):
        return utility(board)
    # initialise min value at above all possible outcomes
    v = 2
    # for all remaining actions, find the min utility given the opponent maximises their utility in the next move
    for action in actions(board):
        v = min(v, maxvalue(result(board, action)))
    return v