import gym

import numpy as np
import time
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")


file_name = 'q_table3.npy'

# Q-learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4000

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

SHOW_EVERY = 1000
STATS_EVERY = 100

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}



# Exploration settings
epsilone = 1
START_EPSILON_DECAYING = 1
END_EPSILONE_DECAYING = EPISODES//2
epsilone_decay_value = epsilone/(END_EPSILONE_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    episode_reward = 0

    # Start new play every episode
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        # ???????????????????????????
        # Just pick a random 0-1 value and compare with epsilone
        if np.random.random() > epsilone:

            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        # pass action and get new state
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # get new_discrete_state
        new_discrete_state = get_discrete_state(new_state)

        #print(reward, new_state, done, _)
        if episode % SHOW_EVERY == 0:
            env.render()

        # if simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible q-value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update q table with new q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achived
        # update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            # q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILONE_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilone -= epsilone_decay_value

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average_reward: {average_reward:>4.1f}, current epsilon: {epsilone:>1.2f}')



for row in q_table:
    print(row)

np.save(file_name, q_table)

for i in range(0,10):
    print(np.random.random())

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()



