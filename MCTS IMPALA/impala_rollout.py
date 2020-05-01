import copy

import gym
import numpy as np
import torch.multiprocessing as mp
from impala.actor import actor
from impala.learner import learner
from impala.model import Network
from impala.parameter_server import ParameterServer
import statistics

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

NUM_ACTORS = 4
ACTOR_TIMEOUT = 500000

class impala_rollout:


    def __init__(self,env):
        self.env = copy.deepcopy(env)
        
        
    def simulate_impala_rollout(self):
    
        nS = np.shape(self.env.observation_space)[0]
        nA = self.env.action_space.n
        queue = mp.Queue()
        reward_queue = mp.Queue()
        #process_manager = mp.Manager()
        #return_dict = process_manager.dict()

        learner_model = Network(nS, nA, "cpu")
        actor_model = Network(nS, nA, "cpu")

        BaseManager.register('ParameterServer', ParameterServer)
        manager = BaseManager()
        manager.start()
        parameter_server = manager.ParameterServer()

        learner_process = mp.Process(target = learner, args=(learner_model,  queue, reward_queue, parameter_server))
        # Currently each actor has its own object via deepcopy. What happens if I don't explicitly do deepcopy?
        actors = [mp.Process(target = actor, args = (copy.deepcopy(actor_model), queue, reward_queue, copy.deepcopy(self.env), parameter_server, i)) for i in range(NUM_ACTORS)]
        print("Actor will now start")
        [actor.start() for actor in actors]
        learner_process.start()
        print("Learner has started")
        [actor.join() for actor in actors]
        print("Rollout Flag 1")
        learner_process.join()
        print("Rollout Flag 2")
        print(statistics.mean(reward_queue.get()))
        return statistics.mean(reward_queue.get())
        



