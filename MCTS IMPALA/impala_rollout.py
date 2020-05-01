import copy
from multiprocessing import Array

import numpy as np
import torch.multiprocessing as mp

from impala.actor import actor
from impala.learner import learner
from impala.model import Network

NUM_ACTORS = 4
ACTOR_TIMEOUT = 500000


class ImpalaRollout:

    def __init__(self, env):
        self.env = copy.deepcopy(env)

    def simulate_impala_rollout(self, learner_model, parameter_server):
        nS = np.shape(self.env.observation_space)[0]
        nA = self.env.action_space.n
        queue = mp.Queue()
        # reward_queue = mp.Queue(maxsize=NUM_ACTORS)
        # process_manager = mp.Manager()
        # return_dict = process_manager.dict()
        mean_rewards = Array('d', [0] * NUM_ACTORS)
        terminated = Array('i', [0] * NUM_ACTORS)

        # learner_model = Network(nS, nA, "cpu")
        actor_model = Network(nS, nA, "cpu")

        # BaseManager.register('ParameterServer', ParameterServer)
        # manager = BaseManager()
        # manager.start()
        # parameter_server = manager.ParameterServer()

        learner_process = mp.Process(target=learner, args=(learner_model, queue, terminated, parameter_server))

        # Currently each actor has its own object via deepcopy. What happens if I don't explicitly do deepcopy?
        actors = [mp.Process(target=actor, args=(
        copy.deepcopy(actor_model), queue, mean_rewards, terminated, copy.deepcopy(self.env), parameter_server, i)) for
                  i in range(NUM_ACTORS)]
        #print("Actor will now start")
        [actor.start() for actor in actors]

        # mean_reward = learner(learner_model, queue, reward_queue, parameter_server)
        learner_process.start()
        # mean_reward = reward_queue.get()
        #print(mean_rewards[:])
        learner_process.join()

        #print(mean_rewards[:])
        # [actor.join() for actor in actors]
        [actor.terminate() for actor in actors]

        queue.close()

        #print(np.mean(mean_rewards))
        return np.mean(mean_rewards)
