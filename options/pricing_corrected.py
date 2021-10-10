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

#######################################################################
# Adapted to American pricing task.                                   # 
# 2020.10 Linhai Qiu (linhaiqiu.ustc@gmail.com)                       #
#######################################################################

import numpy as np
 
# Max exercise time.
T = int(5)
 
# Strike price.
K = 100.0
 
# Initial price upper bound
INIT_PRICE_MAX = 105.0
 
# Inital price lower bound
INIT_PRICE_MIN = 95.0
 
# Delta for KL divergence.
DELTA = 0.1
 
# Probability of price going up.
UP_PROB = 0.5
 
# The price up/down multiplier.
UP_FACTOR = 1.02
DOWN_FACTOR = 0.98
 
 
# Returns the next price if the price goes up.
def up_price(current_price):
    return round(UP_FACTOR * current_price, 2)
 
 
# Returns the next price if the price goes down.
def down_price(current_price):
    return round(DOWN_FACTOR * current_price, 2)
 
 
# Returns a random initial price.
def initial_price():
    return INIT_PRICE_MIN + (INIT_PRICE_MAX - INIT_PRICE_MIN) * np.random.rand()
 
 
# Generates data.
def generate_data(num_traj, p_up):
    all_traj = []
    for n in np.arange(num_traj):
        traj = []
        s = initial_price()
        traj.append(s)
        for t in np.arange(T + 1):
            p = np.random.rand()
            if p < p_up:
                s = up_price(s)
                traj.append(s)
            else:
                s = down_price(s)
                traj.append(s)
        all_traj.append(traj)
    return all_traj
 
 
# Returns the observed price up probability from generated data.
def get_p_hat(all_traj):
    num_up = 0
    num_down = 0
    for traj in all_traj:
        index = 0
        while index < len(traj) - 1:
            if traj[index + 1] > traj[index]:
                num_up = num_up + 1
            else:
                num_down = num_down + 1
            index = index + 1
    return float(num_up) / float(num_up + num_down)
 
 
# Max price.
def max_price():
    s_max = INIT_PRICE_MAX
    for i in np.arange(T):
        s_max = up_price(s_max)
    return s_max
 
 
# Min price.
def min_price():
    s_min = INIT_PRICE_MIN
    for i in np.arange(T):
        s_min = down_price(s_min)
    return s_min
 
 
# Gets price index in state value array.
def get_state_index(s, s_min):
    return int(round((s - s_min) * 100))
 
 
# Gets price from the index of the state value array.
def get_state_from_index(index, s_min):
    return round(s_min + float(index) / 100.0, 2)
 
 
# Min possible price
s_min = min_price()
 
# Max possible price
s_max = max_price()
 
# Max index based on state in the state value array
max_state_index = get_state_index(s_max, s_min)
 
# Generates training data.
num_traj = 10000
all_traj = generate_data(num_traj, UP_PROB)
p_hat =  get_p_hat(all_traj)
 
 
# The inverse second part of Equation (5).
def fn_b(p, b, sv_up, sv_down):
    return b * np.log(p * np.exp(-sv_up/b) + (1-p) * np.exp(-sv_down/b)) + b * DELTA
 
 
def tri_search(p, sv_up, sv_down, b_min = 1e-5, b_max = 1000.):
    f1 = 0
    f2 = 1
    while (np.abs(f1 - f2) > 1e-5):
        b1 = (b_min + b_max) / 2
        b2 = (b1 + b_max) / 2
        f1 = fn_b(p, b1, sv_up, sv_down)
        f2 = fn_b(p, b2, sv_up, sv_down)
        if f1 < f2:
            b_max = b2
        else:
            b_min = b1
 
    return f1
 
 
# Evaluates policies using test trajectories.
def evaluate(policy, p_up):
    n_game = 10000
    total_reward = 0.0
    print("Test price up probability {}".format(p_up))
    for i in range(n_game):
        current_price = initial_price()
        for timestamp in np.arange(T+1):
            action = policy[get_state_index(current_price, s_min)][timestamp]
            if action > 0:
                total_reward += max(0, K - current_price)
                break
            else:
                assert timestamp != T
                rnd = np.random.rand()
                if rnd < p_up:
                    current_price = up_price(current_price)
                else:
                    current_price = down_price(current_price)
 
    print("Average total reward: {}".format(total_reward / float(n_game)))
 
 
# Finds and evaluates policy.
def find_and_evaluate_policy(find_robust_policy):
 
    eval_policy = True
    print_value_iteration_progress = True
    print_value_and_policy_result_details = False
 
    print("Find robust policy? {}".format(find_robust_policy))
    print("s_min: {}, s_max: {}, max_state_index: {}".format(s_min, s_max, max_state_index))
    print("Real p: {}, p_hat: {}".format(UP_PROB, p_hat))
 
    # State value map.
    state_value = np.zeros([max_state_index + 1, T + 1])
    old_state_value = state_value.copy()
 
    # Value iteration.
    num_iterations = 0
    while True:
        num_iterations = num_iterations + 1
        for index in np.arange(max_state_index + 1):
            for timestamp in np.arange(T + 1):
                current_price = get_state_from_index(index, s_min)
                next_up_price = up_price(current_price)
                next_down_price = down_price(current_price)
                action_returns = []
                # Does not exercise.
                # Note that if next price is out of this range or timestamp is T,
                # one must exercise at the current price. Actually, the
                # current price cannot be reached if the next price is out of
                # range while timestamp is not T.
                if timestamp < T and next_up_price <= s_max and next_down_price >= s_min:
                    next_up_price = up_price(current_price)
                    next_down_price = down_price(current_price)
                    sv_up = state_value[get_state_index(next_up_price, s_min), timestamp + 1]
                    sv_down = state_value[get_state_index(next_down_price, s_min), timestamp + 1]
                    action_return = 0
                    if find_robust_policy:
                        action_return = -tri_search(p_hat, sv_up, sv_down)
                    else:
                        action_return = p_hat * sv_up + (1 - p_hat) * sv_down
                    action_returns.append(action_return)
                else:
                    action_returns.append(0)
                # Exercises
                action_return = max(0, K - current_price)
                action_returns.append(action_return)
                # Updates the value of the current_price.
                state_value[get_state_index(current_price, s_min), timestamp] = np.max(action_returns)
        error = np.absolute(state_value - old_state_value).max()
        if print_value_iteration_progress:
            if num_iterations % 2 == 0:
                print("After {} iterations remaining error {}".format(num_iterations, error))
        old_state_value = state_value.copy()
        if error < 1e-5:
            break
    print("value iteration done with {} iterations".format(num_iterations))
 
    # compute the optimal policy
    policy = np.zeros([max_state_index + 1, T + 1])
    for index in np.arange(max_state_index + 1):
        for timestamp in np.arange(T + 1):
            current_price = get_state_from_index(index, s_min)
            # Does not exercise.
            next_up_price = up_price(current_price)
            next_down_price = down_price(current_price)
            action_return_0 = 0.0
            # Note that if next price is out of this range or timestamp is T,
            # one must exercise at the current price. Actually, the
            # current price cannot be reached if the next price is out of
            # range while timestamp is not T.
            if timestamp < T and next_up_price <= s_max and next_down_price >= s_min:
                sv_up = state_value[get_state_index(next_up_price, s_min), timestamp + 1]
                sv_down = state_value[get_state_index(next_down_price, s_min), timestamp + 1]
                if find_robust_policy:
                    action_return_0 = -tri_search(p_hat, sv_up, sv_down)
                else:
                    action_return_0 = p_hat * sv_up + (1 - p_hat) * sv_down
            # Exercises.
            action_return_1 = max(0.0, K - current_price)
            policy[index][timestamp] = 1 if action_return_1 >= action_return_0 else 0
 
    print("Optimal policy extraction done with policy ")
    # The following is detailed information for debugging purposes.
    #  Please comment out when running for larger scale.
    if print_value_and_policy_result_details:
        for index in np.arange(max_state_index + 1):
            for timestamp in np.arange(T + 1):
                current_price = get_state_from_index(index, s_min)
                next_up_price = up_price(current_price)
                next_down_price = down_price(current_price)
                action_return_0 = 0.0
                sv_up = 0.0
                sv_down = 0.0
                if timestamp < T and next_up_price <= s_max and next_down_price >= s_min:
                    sv_up = state_value[get_state_index(next_up_price, s_min), timestamp + 1]
                    sv_down = state_value[get_state_index(next_down_price, s_min), timestamp + 1]
                    action_return_0 = p_hat * sv_up + (1 - p_hat) * sv_down
                    action_return_1 = max(0.0, K - current_price)
                assert index == get_state_index(current_price, s_min)
                print("The value of price {} at index {} time {} is {} with policy {}; next_up {} next_down {}; action0_value {} action1_value {}".format(current_price, index, timestamp, state_value[index, timestamp], policy[index, timestamp], next_up_price, next_down_price, action_return_0, action_return_1))
 
    if eval_policy:
        if find_robust_policy:
            print("The following is the robust policy eval results: ")
        else:
            print("The following is the non-robust policy eval results: ")
        evaluate(policy, UP_PROB + 0.2)
        evaluate(policy, UP_PROB - 0.2)
        print("policy evaluation done!")
        print("\n\n")
 
 
if __name__ == '__main__':
    find_and_evaluate_policy(True)
    find_and_evaluate_policy(False)
