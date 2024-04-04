import copy
import sys

from exceptions import AgentException
from consts import *
from connect4 import Connect4

class MinMaxAgent:
    def __init__(self, my_token):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.minmax(MAX_DEPTH, True, connect4)
    
    def minmax(self, depth, maximizing_player, connect4):
        if depth == 0 or connect4.game_over:
            return self.evaluate_board(connect4)
            
        choice = None
        best_value = None
        comparison_operator = None

        if maximizing_player:
            best_value = -float('inf')
            comparison_operator = max
        else:
            best_value = float('inf')
            comparison_operator = min

        for move in connect4.possible_drops():
            new_connect4 = copy.deepcopy(connect4)
            new_connect4.drop_token(move)
            eval = self.minmax(depth - 1, not maximizing_player, new_connect4)

            best_value = comparison_operator(best_value, eval)
            if eval == best_value:
                choice = move

        return choice

    def evaluate_board(self, connect4):
        score = 0
        opponent_token = 'o' if self.my_token == 'x' else 'x'

        if connect4.wins == self.my_token:
            return sys.maxsize
        if connect4.wins == opponent_token:
            return -sys.maxsize
        
        for line in connect4.iter_fours():
            my_count = line.count(self.my_token)
            opponent_count = line.count(opponent_token)
            if my_count > 0 and opponent_count == 0:
                score += 1
            elif opponent_count > 0 and my_count == 0:
                score -= 1
        return score
