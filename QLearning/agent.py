import argparse
import os
import sys
import pickle
import random
import numpy
from datetime import datetime
from game import Game

argument_parser = argparse.ArgumentParser(description='Q-Learning')
argument_parser.add_argument('--qtable', type=str, default=None)

args = argument_parser.parse_args()

ALPHA = 0.2
GAMMA = 0.5
EPSILON = 0.15

MAX_ITERATIONS = 10000

if __name__ == '__main__':
    try:
        starting_iteration = 1

        if args.qtable:
            with open(args.qtable, 'rb') as f:
                q_table = pickle.load(f)
                print("Q-Table loaded successfully")
        else:
            q_table = {}

        max_achieved_score = 0
        environment = Game()
        environment.start()

        for i in range(starting_iteration, MAX_ITERATIONS):

            state = environment.restart()
            reward = 0
            game_over = False

            while not game_over:

                if state % 2 == 1:
                    action = 0
                elif environment.get_score() >= max_achieved_score and random.uniform(0, 1) < EPSILON:
                    action = random.randint(0, 1)                               # Exploration
                else:
                    action = numpy.argmax(q_table.get(state, [0, 0]))           # Exploitation

                next_state, reward, game_over = environment.perform_action(action)

                try:
                    old_value = q_table.get(state, [0, 0])[action]
                except:
                    old_value = 0

                next_max = numpy.max(q_table.get(next_state, [0, 0]))
                new_value = ((1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max))
                new_value = numpy.float16(new_value)

                try:
                    q_table[state][action] = new_value
                except:
                    new_arr = numpy.array([0, 0], dtype=numpy.float16)
                    new_arr[action] = new_value
                    q_table[state] = new_arr

                state = next_state

            final_score = environment.get_score()
            print(f"TIME: {datetime.now()} | ITERATION: {i} | SCORE: {final_score}")
            max_achieved_score = (final_score // 100) * 100

            if i % 10 == 0:
                max_achieved_score += 100

            if i % 100 == 0:
                with open(f'new_qtable_iteration_{i}.pickle', 'wb') as f:
                    pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)


    finally:
        with open(f'interrupted_qtable.pickle', 'wb') as f:
            pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

        environment.quit()