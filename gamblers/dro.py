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

def fn_b(p, b, svp, svn, radius=0.1):
    return b * np.log(p * np.exp(-svp/b) + (1-p) * np.exp(-svn/b)) + b * radius


def tri_search(p, svp, svn, radius=0.1, b_max=10.):
    b_min = 1e-3
    f1 = 0
    f2 = 1
    # while (b_min + 1e-3 < b_max):
    while (np.abs(f1-f2) > 1e-5):
        b1 = (b_min + b_max) / 2
        b2 = (b1 + b_max) / 2
        f1 = fn_b(p, b1, svp, svn, radius=radius)
        f2 = fn_b(p, b2, svp, svn, radius=radius)
        if f1 < f2:
            b_max = b2
        else:
            b_min = b1

    return f1


def newton_grad(p, b, svp, svn, radius=0.02):
    """
    Args:
        p: probability of head up
        b: beta
        svp: state_value[state + action]
        svn: state_value[state - action]
        radius: radius of robust set
    
    Returns:
        f', f": 1st and 2nd deriv for Newton update
    """

    esvp = np.exp(-svp/b)
    esvn = np.exp(-svn/b)
    pesv = p * esvp + (1-p) * esvn
    log_term = np.log(pesv)
    f_b = b * log_term + b * radius

    dpesv = p * esvp * svp / b**2 + (1-p) * esvn * svn / b**2 
    f_1 = log_term + radius + b * dpesv / pesv

    dpesv2 = p * esvp * svp**2 / b**4 + (1-p) * esvn * svn**2 / b**4
    term_1 = dpesv / pesv
    term_2 = dpesv * (pesv - b * dpesv) / pesv**2
    term_3 = (dpesv2 * b - 2 * dpesv) / pesv
    f_2 = term_1 + term_2 + term_3 
    return f_b, f_1, f_2


def newton(p, b, svp, svn, radius=0.02, epsilon=1e-5, b_min=1e-6):
    count = 0
    while(True):
        count = count + 1
        f_b, f_1, f_2 = newton_grad(p, b, svp, svn, radius=radius)
        if np.abs(f_1) <= epsilon:
            return f_b, f_2
        else:
            if np.abs(f_2) < epsilon:
                return fn_b(p, b-f_1, svp, svn, radius=radius), f_2
            else:
                b = np.maximum(b_min, b - f_1 / f_2)
        

def figure_4_3():

    eval_policy = True
    plot_policy = False

    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    beta0 = 10.
    radius = 0.02
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                # action_returns.append(
                #     HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

                # DRO
                svp = state_value[state + action]
                svn = state_value[state - action]
                sv_std = HEAD_PROB * svp + (1 - HEAD_PROB) * svn
                # sv_rob, hess = newton(HEAD_PROB, 
                #                       beta0, 
                #                       state_value[state + action], 
                #                       state_value[state - action],
                #                       radius=radius)
                # if np.abs(hess) < 1e-6:
                #     action_return = sv_std
                # else:
                #     print("non-zero 2nd deriv")
                #     import pdb; pdb.set_trace()
                #     action_return = -sv_rob

                if svp + svn <= 0:
                    action_return = sv_std
                else:
                    action_return = -tri_search(HEAD_PROB, svp, svn, radius=radius)
                action_returns.append(action_return)

            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-5:
            sweeps_history.append(state_value)
            break

    # print("value iteration done!")

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

    # print("optimal policy extraction done!")

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

        plt.savefig('dro_images/figure_4_3.png')
        plt.close()


def evaluate(policy):
    n_game = 10000
    init_state = 50
    Goal = 100
    p_head = 0.5
    n_win = 0
    import pdb; pdb.set_trace()
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
