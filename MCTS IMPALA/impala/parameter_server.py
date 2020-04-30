import torch.multiprocessing as mp


class ParameterServer:
    def __init__(self):
        self.lock = mp.Lock()
        self.weights = None

    def push(self, weights):
        with self.lock:
            self.weights = weights

    def pull(self):
        with self.lock:
            return self.weights
