from fileinput import filename
import sys
import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env
import matplotlib.pyplot as plt


MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def doorkey_problem(env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq

def train_control_policy(env, info, num):

    cost_grid = create_cost_grid_B(env, info)   # width x height x pose x key x door

    # cost_grid = update_current_step(env, cost_grid, [3, 3])

    start_x = info['init_agent_pos'][0]
    start_y = info['init_agent_pos'][1]

    # robot orientation
    if np.array_equal(info['init_agent_dir'], np.array([1, 0])):
        p = 0
    elif np.array_equal(info['init_agent_dir'], np.array([0, -1])):
        p = 1
    elif np.array_equal(info['init_agent_dir'], np.array([-1, 0])):
        p = 2
    elif np.array_equal(info['init_agent_dir'], np.array([0, 1])):
        p = 3

    # robot does not have the key 
    k = 0

    # door status (d = 0: unlock, d = 1: lock)
    if info['door_open'][0]:
        d1 = 0
    else:
        d1 = 1

    if info['door_open'][1]:
        d2 = 0
    else:
        d2 = 1
    
    paths = []

    cnt = 0
    # iterate the entire maze
    while cost_grid[start_x][start_y][p][k][d1][d2] == np.inf:
        for x in range(info['width']):
            for y in range(info['height']):
                cost_grid = update_current_step(env, cost_grid, [x, y], paths)
                cnt = cnt + 1


    # print("total cost: {} energy points".format(cost_grid[start_x][start_y][p][k][d1][d2]))
    # print("# of iterations: {}".format(cnt))
    if num < 10:
        file_name = f'cost-grid-0{num}.npy'
    else:
        file_name = f'cost-grid-{num}.npy'

    np.save(file_name, cost_grid)

    # # get the path
    # queue = find_path(paths, start_x, start_y, p, k, d1, d2)
    # actions = translate_to_actions(queue)

    # print("actions: {}".format(actions))

    pass

def partA():
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    seq = doorkey_problem(env) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save
    
def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)    

def training_stage():

    # env_folder = './envs/random_envs'
    # env, info, env_path = load_random_env(env_folder)

    for num in range(1, 37):
        if num < 10:
            env_path = f'./envs/random_envs/DoorKey-8x8_0{num}.pickle'
        else:
            env_path = f'./envs/random_envs/DoorKey-8x8_{num}.pickle'
        env, info, env_path = load_training_data(env_path)

        train_control_policy(env, info, num)

    pass

def find_the_next_input(cost_grid, ptr_x, ptr_y, ptr_p, ptr_k, ptr_d1, ptr_d2):

    input = -1

    candidates = np.zeros((6, 1))

    # MF
    if ptr_p == 0:
        if ptr_x + 1 < 8:
            candidates[0] = cost_grid[ptr_x + 1][ptr_y][ptr_p][ptr_k][ptr_d1][ptr_d2]
        else:
            candidates[0] = np.inf

    elif ptr_p == 1:
        if ptr_y > 0:
            candidates[0] = cost_grid[ptr_x][ptr_y - 1][ptr_p][ptr_k][ptr_d1][ptr_d2]
        else:
            candidates[0] = np.inf
    elif ptr_p == 2:
        if ptr_x > 0:
            candidates[0] = cost_grid[ptr_x - 1][ptr_y][ptr_p][ptr_k][ptr_d1][ptr_d2]
        else:
            candidates[0] = np.inf    
    elif ptr_p == 3:
        if ptr_y + 1 < 8:
            candidates[0] = cost_grid[ptr_x][ptr_y + 1][ptr_p][ptr_k][ptr_d1][ptr_d2]
        else:
            candidates[0] = np.inf

    # TL
    candidates[1] = cost_grid[ptr_x][ptr_y][(ptr_p + 1) % 4][ptr_k][ptr_d1][ptr_d2]
    # TR
    candidates[2] = cost_grid[ptr_x][ptr_y][(ptr_p - 1) % 4][ptr_k][ptr_d1][ptr_d2]
    # PK
    candidates[3] = cost_grid[ptr_x][ptr_y][ptr_p][1][ptr_d1][ptr_d2]
    # UD 1
    candidates[4] = cost_grid[ptr_x][ptr_y][ptr_p][ptr_k][0][ptr_d2]
    # UD 2
    candidates[5] = cost_grid[ptr_x][ptr_y][ptr_p][ptr_k][ptr_d1][0]

    input = np.where(candidates == np.amin(candidates))[0][0]

    return input

def cal_policy_table(num):

    time_horizon = 20
    control_policy_table = np.zeros((8, 8, 4, 2, 2, 2, time_horizon))    # width x height x pose x key x door1 x door2 x time


    # load the cost grid
    if num < 10:
        cost_grid = np.load(f'cost-grid-0{num}.npy')
    else:
        cost_grid = np.load(f'cost-grid-{num}.npy')

    # for idx in range(36):
    #     num = idx + 1
    #     if num < 10:
    #         file_name = f'cost-grid-0{num}.npy'
    #     else:
    #         file_name = f'cost-grid-{num}.npy'
    #     cost_grid = np.load(file_name)

    itr = 0

    # find the control 
    while itr < time_horizon:
        for ptr_x in range(8):
            for ptr_y in range(8):
                for ptr_p in range(4):
                    for ptr_k in range(2):
                        for ptr_d1 in range(2):
                            for ptr_d2 in range(2):
                                if itr < time_horizon:
                                    control_policy_table[ptr_x][ptr_y][ptr_p][ptr_k][ptr_d1][ptr_d2][itr] = find_the_next_input(cost_grid, ptr_x, ptr_y, ptr_p, ptr_k, ptr_d1, ptr_d2)
                                else:
                                    break
        itr = itr + 1

    
    if num < 10:
        np.save(f'control_policy_table_0{num}.npy', control_policy_table)
    else:
        np.save(f'control_policy_table_{num}.npy', control_policy_table)
    return control_policy_table

def test_control_policy(env, info):

    generic_optimal_control_policy_table = np.load('generic_optimal_control_policy_table.npy')

    # the robot always starts at the same position with the same orientation
    cur_x = 3
    cur_y = 5
    cur_p = 1
    cur_k = 0
    cur_d1 = 0
    cur_d2 = 0

    # the environment
    # goal's position
    if np.array_equal(info['goal_pos'], [5, 1]):
        goal_pos = 0
    elif np.array_equal(info['goal_pos'], [6, 3]):
        goal_pos = 1
    elif np.array_equal(info['goal_pos'], [5, 6]):
        goal_pos = 2

    # key's position
    if np.array_equal(info['key_pos'], [1, 1]):
        key_pos = 0
    elif np.array_equal(info['key_pos'], [2, 3]):
        key_pos = 1
    elif np.array_equal(info['key_pos'], [1, 6]):
        key_pos = 2

    # door1's status
    d1_pos = 0 if info['door_open'][0] else 1

    # door2's status
    d2_pos = 0 if info['door_open'][1] else 1

    goal_x = info['goal_pos'][0]
    goal_y = info['goal_pos'][1]

    t = 0
    actions = []
    while cur_x != goal_x or cur_y != goal_y:

        # find the control 
        # width x height x pose x key x door1 x door2 x time
        # (width x height x pose x key x door1 x door2) x time horizon x (key position x door1 x door2 x goal)
        # generic_optimal_control_policy_table[:, :, :, :, :, :, :, key_pos, d1_pos, d2_pos, goal_pos] = control_policy
        action = generic_optimal_control_policy_table[cur_x][cur_y][cur_p][cur_k][cur_d1][cur_d2][t][key_pos][d1_pos][d2_pos][goal_pos]
        
        actions.append(action)

        # MF
        if action == 0:
            if cur_p == 0:
                cur_x = cur_x + 1
            elif cur_p == 1:
                cur_y = cur_y - 1
            elif cur_p == 2:
                cur_x = cur_x - 1
            elif cur_p == 3:
                cur_y = cur_y + 1
        # TL
        elif action == 1:
            cur_p = (cur_p + 1) % 4
        # TR
        elif action == 2:
            cur_p = (cur_p - 1) % 4
        # PK
        elif action == 3:
            cur_k = 1
        # UD 1
        elif action == 4:
            cur_d1 = 0
        # UD 2
        elif action == 5:
            cur_d2 = 0

        t = t + 1
    return actions

def plot_random_maze():

    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    plt.plot(env)

def maze_information():

    maze_info = np.zeros((36, 4))   # key position x door1 x door2 x goal

    for idx in range(36):
        num = idx + 1
        if num < 10:
            env_path = f'./envs/random_envs/DoorKey-8x8_0{num}.pickle'
        else:
            env_path = f'./envs/random_envs/DoorKey-8x8_{num}.pickle'
        _, info, _ = load_training_data(env_path)
        
        # key's position
        if np.array_equal(info['key_pos'], [1, 1]):
            maze_info[idx][0] = 0
        elif np.array_equal(info['key_pos'], [2, 3]):
            maze_info[idx][0] = 1
        elif np.array_equal(info['key_pos'], [1, 6]):
            maze_info[idx][0] = 2

        # door1's status
        if info['door_open'][0]:
            maze_info[idx][1] = 0
        else:
            maze_info[idx][1] = 1   # lock

        # door1's status
        if info['door_open'][1]:
            maze_info[idx][2] = 0
        else:
            maze_info[idx][2] = 1   # lock
        
        # goal's position
        if np.array_equal(info['goal_pos'], [5, 1]):
            maze_info[idx][3] = 0
        elif np.array_equal(info['goal_pos'], [6, 3]):
            maze_info[idx][3] = 1
        elif np.array_equal(info['goal_pos'], [5, 6]):
            maze_info[idx][3] = 2

    # store the info
    np.save('maze_info.npy', maze_info)

    pass

def generic_optimal_control_policy():

    # (width x height x pose x key x door1 x door2) x time horizon x (key position x door1 x door2 x goal)

    generic_optimal_control_policy_table = np.zeros((8, 8, 4, 2, 2, 2, 20, 3, 2, 2, 3))   

    maze_info = np.load('maze_info.npy')

    for idx in range(36):
        num = idx + 1
        if num < 10:
            control_policy = np.load(f'control_policy_table_0{num}.npy')
        else:
            control_policy = np.load(f'control_policy_table_{num}.npy')

        # replace the according portion
        key_pos  = int(maze_info[idx][0]) 
        d1_pos   = int(maze_info[idx][1]) 
        d2_pos   = int(maze_info[idx][2]) 
        goal_pos = int(maze_info[idx][3]) 
        generic_optimal_control_policy_table[:, :, :, :, :, :, :, key_pos, d1_pos, d2_pos, goal_pos] = control_policy
    
    # save the generic_optimal_control_policy_table
    np.save('generic_optimal_control_policy_table.npy', generic_optimal_control_policy_table)

    pass

if __name__ == '__main__':

    # select which part would the user like to run
    # part a: 0
    # part b: 1
    part = 0
    if part == 0:
        partA()
    else:
        partB()
    
    # training_stage()  # completed
    # for idx in range(36):
    #     num = idx + 1
    #     control_policy_table = cal_policy_table(num)
    # maze_information()
    # generic_optimal_control_policy()
    # test_control_policy()
    partB()