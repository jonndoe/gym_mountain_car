import gym
import time
import numpy as np
import random


env = gym.make("MountainCar-v0")
state = env.reset()

done = False

q_table = np.load('q_table.npy')
for row in q_table:
    print(row)

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

first_discrete_state = get_discrete_state(env.reset())
print('first_discrete_state:', first_discrete_state)
print('q_values of f_d_s:', q_table[first_discrete_state])

counter = 0

# default current state
current_discrete_state = q_table[first_discrete_state]

while not done:

    if counter == 0:
        # get first action from q_table
        action = np.argmax(q_table[first_discrete_state])
    else:
        # get action based on current_discrete_state
        action = np.argmax(q_table[current_discrete_state])

    # get current new_state of environment and convert it to discrete state
    new_state, reward, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)

    # update new_state variable for next loop
    current_discrete_state = new_discrete_state


    env.render()
    time.sleep(0.05)

    counter += 1
    print(counter)
