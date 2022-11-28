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

def my_solution_A(env, info):

    cost_grid = create_cost_grid(env, info)   # width x height x pose x key x door

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
    if info['door_open']:
        d = 0
    else:
        d = 1

    paths = []

    cnt = 0
    # iterate the entire maze
    while cost_grid[start_x][start_y][p][k][d] == np.inf:
        for x in range(info['width']):
            for y in range(info['height']):
                cost_grid = update_current_step(env, cost_grid, [x, y], paths)
                cnt = cnt + 1


    print("total cost: {} energy points".format(cost_grid[start_x][start_y][p][k][d]))
    print("# of iterations: {}".format(cnt))

    # get the path
    queue = find_path(paths, start_x, start_y, p, k, d)
    actions = translate_to_actions(queue)

    print("actions: {}".format(actions))

    return actions

def partA():
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    seq = doorkey_problem(env) # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save

def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)

def my_work_A():
    '''
    doorkey-5x5-normal.env

    doorkey-6x6-direct.env
    doorkey-6x6-normal.env
    doorkey-6x6-shortcut.env

    doorkey-8x8-direct.env
    doorkey-8x8-normal.env
    doorkey-8x8-shortcut.env
    '''

    maze = 6
    # select the according maze
    if maze == 0:
        env_path = './envs/doorkey-5x5-normal.env'
    elif maze == 1:
        env_path = './envs/doorkey-6x6-direct.env'
    elif maze == 2:
        env_path = './envs/doorkey-6x6-normal.env'
    elif maze == 3:
        env_path = './envs/doorkey-6x6-shortcut.env'
    elif maze == 4:
        env_path = './envs/doorkey-8x8-direct.env'
    elif maze == 5:
        env_path = './envs/doorkey-8x8-normal.env'
    elif maze == 6:
        env_path = './envs/doorkey-8x8-shortcut.env'

    env, info = load_env(env_path) # load an environment
    seq = my_solution_A(env, info)

def my_work_B():

    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    print(info)

    seq = my_solution_B(env, info)

    pass

def plot_random_maze():

    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)
    plt.plot(env)


if __name__ == '__main__':
    # example_use_of_gym_env()
    # partA()
    # partB()
    my_work_A()
