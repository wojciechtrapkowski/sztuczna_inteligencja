from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent

amount_of_random_agent_wins = 0
amount_of_minmax_agent_wins = 0
amount_of_draws = 0

for i in range(10):
    connect4 = Connect4(width=7, height=6)
    agent1 = RandomAgent('o')
    agent2 = MinMaxAgent('x')
    while not connect4.game_over:
        connect4.draw()
        try:
            if connect4.who_moves == agent1.my_token:
                n_column = agent1.decide(connect4)
            else:
                n_column = agent2.decide(connect4)
            connect4.drop_token(n_column)
        except (ValueError, GameplayException):
            print('invalid move')
    
    connect4.draw()
    if connect4.wins == 'x':
        amount_of_minmax_agent_wins += 1
    elif connect4.wins == 'o':
        amount_of_random_agent_wins += 1
    else:
        amount_of_draws += 1

print(f"{amount_of_minmax_agent_wins} - {amount_of_random_agent_wins} - {amount_of_draws}")
