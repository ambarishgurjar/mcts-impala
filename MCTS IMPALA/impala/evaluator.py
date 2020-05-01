import gym
import numpy as np
from impala.model import Network
import torch as T


def evaluator(initial_weights):
    env = gym.make('CartPole-v1')
    nS = np.shape(env.observation_space)[0]
    nA = env.action_space.n
    model = Network(nS, nA, "cpu")
    model.load_state_dict(initial_weights)
    done = False
    current_state = env.reset()
    total_reward = 0
    hx = T.zeros((2, 1, 8))
    while not done:
        action, _, hx = model(current_state, T.tensor(done, dtype=T.bool).view(1,1), hx, actor=True)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        current_state = next_state
    return total_reward
