import copy

import gym
import numpy as np
import torch.multiprocessing as mp
from impala.actor import actor
from impala.learner import learner
from impala.model import Network
from impala.parameter_server import ParameterServer

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

NUM_ACTORS = 4
ACTOR_TIMEOUT = 500000

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.seed(42)
    nS = np.shape(env.observation_space)[0]
    nA = env.action_space.n

    queue = mp.Queue()

    learner_model = Network(nS, nA, "cpu")
    actor_model = Network(nS, nA, "cpu")

    BaseManager.register('ParameterServer', ParameterServer)
    manager = BaseManager()
    manager.start()
    parameter_server = manager.ParameterServer()

    learner = mp.Process(target = learner, args=(learner_model, queue, parameter_server))
    # Currently each actor has its own object via deepcopy. What happens if I don't explicitly do deepcopy?
    actors = [mp.Process(target = actor, args = (copy.deepcopy(actor_model), queue, copy.deepcopy(env), parameter_server)) for i in range(NUM_ACTORS)]

    [actor.start() for actor in actors]
    learner.start()
    [actor.join() for actor in actors]
    learner.join()


