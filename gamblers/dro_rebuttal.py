#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Adapted to a distributional robust value iteration version.         # 
# 2020.9 Qinxun Bai (qinxun.bai@gmail.com)                            #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

matplotlib.use('Agg')

# goal
GOAL = 200

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.65

radius = 0.02

def generate_data(n_train, p_head):
    train_data = []
    for n in np.arange(n_train):
        p = np.random.rand()
        if p <= p_head:
            data = 1.
        else:
            data = 0.
        train_data.append(data)
    return train_data


def est_p_hat(train_data):
    return sum(train_data) / len(train_data)


def generate_p_hat(n_train, p_head):
    train_data = generate_data(n_train, p_head)
    return est_p_hat(train_data)


def fn_b(p, b, svp, svn):
    return b * np.log(p * np.exp(-svp/b) + (1-p) * np.exp(-svn/b)) + b * radius


def tri_search(p, svp, svn, b_max=100.):
    b_min = 1e-3
    f1 = 0
    f2 = 1
    # while (b_min + 1e-3 < b_max):
    niter = 0
    while (np.abs(f1-f2) > 1e-5) and (niter < 1e+3):
        b1 = (b_min + b_max) / 2
        b2 = (b1 + b_max) / 2
        f1 = fn_b(p, b1, svp, svn)
        f2 = fn_b(p, b2, svp, svn)
        if f1 < f2:
            b_max = b2
        else:
            b_min = b1
        niter += niter
        
    return f1


def value_iteration(p_head):
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    niter = 0
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                # DRO
                svp = state_value[state + action]
                svn = state_value[state - action]
                sv_std = p_head * svp + (1 - p_head) * svn

                if svp + svn <= 0:
                    action_return = sv_std
                else:
                    action_return = -tri_search(p_head, svp, svn)
                action_returns.append(action_return)

            new_value = np.max(action_returns)
            state_value[state] = new_value

        niter = niter + 1
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-5 or niter > 1e+3:
            sweeps_history.append(state_value)
            break
        
    print("value iteration done!")
    return state_value


def policy_from_state_value(state_value, p_head):
    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            svp = state_value[state + action]
            svn = state_value[state - action]
            sv_std = p_head * svp + (1 - p_head) * svn
            if svp + svn <= 0:
                action_return = sv_std
            else:
                action_return = -tri_search(p_head, svp, svn)
            action_returns.append(action_return)

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    print("optimal policy extraction done!")
    return policy


def policy_eval(policy, p_head):
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0
    sweeps_history = []

    niter = 0
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)
        
        for state in STATES[1:GOAL]:
            action = int(policy[state])
            # DRO
            svp = state_value[state + action]
            svn = state_value[state - action]
            sv_std = p_head * svp + (1 - p_head) * svn

            if svp + svn <= 0:
                action_return = sv_std
            else:
                action_return = -tri_search(p_head, svp, svn)
            state_value[state] = action_return

        niter = niter + 1
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-5 or niter > 1e+3:
            sweeps_history.append(state_value)
            break

    return state_value


def v_hat_from_p_hat(p_hat, p_head):
    state_value_hat = value_iteration(p_hat)
    policy_hat = policy_from_state_value(state_value_hat, p_hat)
    v_hat =  policy_eval(policy_hat, p_head)

    return v_hat


def regret(niter):
    save_dir = "rebuttal"

    # compute optimal state value
    optimal_value = value_iteration(HEAD_PROB)
    filename = "dro_regret4/optimal_value_{}.npy".format(HEAD_PROB)
    np.save(filename, optimal_value)
    print("optimal value saved!")

    ## generate p_hat
    n_train = 100
    n_trial = 20
    for i in range(niter):
        values = []
        p_hats = []
        for j in range(n_trial):
            p_hat = generate_p_hat(n_train, HEAD_PROB)
            p_hats.append(p_hat)
            # value = v_hat_from_p_hat(p_hat, HEAD_PROB)
            # values.append(value)
        p_hats = np.asarray(p_hats)
        print("p_hat_{}: ".format(n_train))
        print(p_hats)
        print("----------------------------------------------------")
        p_file = save_dir + "/p_hats_{}.npy".format(n_train)
        np.save(p_file, p_hats)
        n_train = n_train * 2

    ## load p_hats and compute v_hats
    n_train = 100
    n_train = n_train * niter
    values = []
    p_file = save_dir + "/p_hats_{}.npy".format(n_train)
    p_hats = np.load(p_file)
    for i in range(len(p_hats)):
        value = v_hat_from_p_hat(p_hats[i], HEAD_PROB)
        values.append(value)
        print("{}th value_hat done!".format(i))
    v_hat = np.stack(values)
    v_file = save_dir + "/v_hats_{}.npy".format(n_train)
    np.save(v_file, v_hat)



if __name__ == '__main__':
    niter = int(sys.argv[1])
    regret(niter)
