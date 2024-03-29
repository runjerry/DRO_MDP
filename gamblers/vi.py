#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

""" Adapted to a distributional robust value iteration version. """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.6


def figure_4_3():

    eval_policy = True
    plot_policy = False

    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    # save policy
    filename = "vi_policy/policy_{}.npy".format(HEAD_PROB)
    np.save(filename, policy)

    if eval_policy:
        evaluate(policy)
        print("policy evaluation done!")

    if plot_policy:
        plt.figure(figsize=(10, 20))

        plt.subplot(2, 1, 1)
        for sweep, state_value in enumerate(sweeps_history):
            plt.plot(state_value, label='sweep {}'.format(sweep))
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.scatter(STATES, policy)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')

        plt.savefig('vi_images/figure_4_3.png')
        plt.close()


def evaluate(policy):
    n_game = 10000
    init_state = 50
    Goal = 100
    p_head = 0.55
    n_win = 0
    for i in range(10):
        for j in range(1, 100):
            # state = init_state
            state = j
            while True:
                stake = policy[state]
                rnd = np.random.rand()
                if rnd <= p_head:
                    state += stake
                else:
                    state -= stake
                if state >= Goal:
                    n_win += 1
                    break
                if state <= 0:
                    break
                state = int(state)

    print("num of winning game: {}".format(n_win))



if __name__ == '__main__':
    figure_4_3()
