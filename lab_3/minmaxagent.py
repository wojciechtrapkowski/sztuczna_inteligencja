import copy

from exceptions import AgentException
from consts import *

class MinMaxAgent:
    def __init__(self, my_token, connect4):
        self.my_token = my_token
        self.connect4 = connect4

    def decide(self):
        if self.connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.minmax(MAX_DEPTH, True, self.connect4)
    
    def minmax(self, depth, maximizing_player, connect4):
        if depth == 0 or connect4.game_over:
            return self.evaluate_board(connect4)
            
        choice = -float('inf')

        if maximizing_player:
            max_eval = -float('inf')
            for move in connect4.possible_drops():
                new_connect4 = copy.deepcopy(connect4)
                new_connect4.drop_token(move)
                eval = self.minmax(depth - 1, False, new_connect4)
                if eval > max_eval:
                    max_eval = eval
                    choice = move
            return choice
        else:
            min_eval = float('inf')
            for move in connect4.possible_drops():
                new_connect4 = copy.deepcopy(connect4)
                new_connect4.drop_token(move)
                eval = self.minmax(depth - 1, True, new_connect4)
                if eval < min_eval:
                    min_eval = eval
                    choice = move
            return choice
    
    def undo_move(self, n_column):
         for n_row in range(self.connect4.height - 1, -1, -1):
            if self.connect4.board[n_row][n_column] != '_':
                self.connect4.board[n_row][n_column] = '_'
                self.connect4.who_moves = 'o' if self.connect4.who_moves == 'x' else 'x'
                self.connect4.game_over = False
                return

    def evaluate_board(self, connect4):
        score = 0

        # Evaluate rows
        for row in connect4.board:
            score += self._evaluate_line(row)

        # Evaluate columns
        for col in range(connect4.width):
            column = [connect4.board[row][col] for row in range(connect4.height)]
            score += self._evaluate_line(column)

        # Evaluate diagonals
        for i in range(connect4.height - 3):
            for j in range(connect4.width - 3):
                diagonal1 = [connect4.board[i+k][j+k] for k in range(4)]
                score += self._evaluate_line(diagonal1)
                diagonal2 = [connect4.board[i+k][j+3-k] for k in range(4)]
                score += self._evaluate_line(diagonal2)

        return score

    def _evaluate_line(self, line):
        my_tokens = line.count(self.my_token)
        opponent_tokens = line.count('o')  # 'o' is the opponent's token

        if my_tokens == 4:
            return 1000
        elif opponent_tokens == 4:
            return -1000
        elif my_tokens == 3 and line.count('_') == 1:
            return 100
        elif opponent_tokens == 3 and line.count('_') == 1:
            return -100
        elif my_tokens == 2 and line.count('_') == 2:
            return 10
        elif opponent_tokens == 2 and line.count('_') == 2:
            return -10
        else:
            return 0