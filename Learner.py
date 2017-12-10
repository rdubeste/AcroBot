import gym
import numpy as np
import PolicyAgent as P
import QAgent as Q
import time


# compute the discounted rewards
def discount_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_total = 0
    for index in reversed(range(0, r.size)):
        running_total = running_total * gamma + r[index]
        discounted_r[index] = running_total
    return discounted_r


def calc_reward(state, threshold, penalty):
    # state: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    # height formula (-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.
    # transformation cos (x + y) = cos x cos y − sin x sin y
    # x = s1, y = s0
    # cos (s1 + s0) = (cos s1 cos s0) - (sin s1 sin s0)
    cos0 = state[0]
    sin0 = state[1]
    cos1 = state[2]
    sin1 = state[3]
    # calculate height of the tail end of the pendulum
    height = -cos0 - ((cos1 * cos0) - (sin1 * sin0))
    # height = -height * height



    # calculate average rotational speed of the two axes
    a_speed = np.abs(state[4]) + np.abs(state[5]) / 2

    reward = 0
    if height < -1:
        reward = min(a_speed, 10)
    elif height > 1:
        if a_speed < 2:
            reward = 500
        else:
            reward = 100
    else:
        reward = height + 20
    # if speed is greater than threshold, apply a penalty to the reward
    if a_speed > threshold:
        return height - penalty
    return -reward

# initialize values
env = gym.make('Acrobot-v1')

# agent = P.PolicyAgent(rate=0.01)
agent = P.PolicyAgent(0.01)

episodes = 250
episode_length = 500

reward_history = []
state = None
for i in range(0, episodes):
    state = env.reset()
    running_reward = 0
    episode_history = []
    for t in range(0, episode_length):
        # choose and perform action probabilistically based on agent output
        action_vals = agent.get_output(state)
        action = np.random.choice([0, 1, 2], p=action_vals[0])
        new_state, reward, done, info = env.step(action)
        reward = calc_reward(new_state, 30, 200)
        # np.abs(state[4]) + 1
        # record previous state, action, reward, and new state
        episode_history.append([state, action, reward, new_state])
        state = new_state
        running_reward += reward
        #env.render()

    # now we process the history and update the array
    history_array = np.array(episode_history)
    states = np.vstack(history_array[:, 0])
    actions = history_array[:, 1]
    rewards = discount_reward(history_array[:, 2], 0.99)
    # add gradients to the buffer
    gradients = agent.compute_gradients(states, actions, rewards)
    agent.train(gradients)
    reward_history.append(running_reward)
    print(i, running_reward)
    if i % 100 == 0 and i != 0:
        print("average:", sum(reward_history[-100:]) / 100)

while True:
    env.reset()
    for t in range(0, episode_length):
        # choose and perform action probabilistically based on agent output
        action_vals = agent.get_output(state)
        action = np.random.choice([0, 1, 2], p=action_vals[0])
        new_state, reward, done, info = env.step(action)
        reward = calc_reward(new_state, 14, 200)
        # np.abs(state[4]) + 1
        # record previous state, action, reward, and new state
        state = new_state
        env.render()
        print(reward)
        time.sleep(0.01)
