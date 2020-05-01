import torch as T

ACTOR_TIMEOUT = 10
STEPS_PER_TRAJECTORY = 100


class Trajectory:
    def __init__(self):
        self.actions = []
        self.rewards = []
        # dones is required to calculate discounts in learner
        self.dones = []
        self.states = []
        self.action_distributions = []
        self.lstm_hx = None

    def append(self, state, action, reward, done, action_distribution):
        self.states.append(self._to_tensor(state, dtype=T.float32, shape=(1, 4)))
        self.actions.append(self._to_tensor(action))
        self.rewards.append(self._to_tensor(reward, dtype=T.float32))
        self.dones.append(self._to_tensor(done, dtype=T.bool))
        self.action_distributions.append(action_distribution)

    def get_last(self):
        return self.states[-1], self.actions[-1], self.rewards[-1], self.dones[-1], self.action_distributions[-1]

    @staticmethod
    def _to_tensor(value, dtype=T.float32, shape=(1, 1)):
        return T.tensor(value, dtype=dtype).view(shape)

    def finish(self):
        """
        states: Tensor of dimension n*1*4 - Vertically stacked
        actions: Tensor of dimension 1*n - Concat
        rewards : Tensor of dimension 1*n  - Concat
        dones : Tensor of dimension 1*n - Concat
        action_distributions: Tensor of dimension n*1*2 - Concat
        """
        self.states = T.stack(self.states)
        self.actions = T.cat(self.actions).squeeze()
        self.rewards = T.cat(self.rewards).squeeze()
        self.dones = T.cat(self.dones).squeeze()
        self.action_distributions = T.cat(self.action_distributions)

    @property
    def length(self):
        return len(self.rewards)


def actor(actor_model, queue, mean_rewards, terminated, env, parameter_server, p_id):
    num_episodes = 1
    cumulative_reward = 0
    steps = 0
    current_state = env.reset()
    hx = T.zeros((2, 1, 8))
    last_state = None

    while steps <= 5000:
        # Pull weights from leaner
        updated_weights = parameter_server.pull()
        if updated_weights is not None:
            actor_model.load_state_dict(updated_weights)

        # Initialize a new trajectory
        trajectory = Trajectory()
        trajectory.lstm_hx = hx.squeeze()
        done = True
        if last_state is not None:
            trajectory.append(*last_state)

        with T.no_grad():
            while True:
                action_to_take, action_distribution, hx = actor_model(current_state, T.tensor(done, dtype=T.bool).view(1,1), hx, actor=True)
                next_state, reward, done, _ = env.step(action_to_take)
                trajectory.append(current_state, action_to_take, reward, done, action_distribution.detach())
                cumulative_reward += reward

                if done:
                    num_episodes += 1
                    next_state = env.reset()
                    hx = T.zeros((2,1,8))

                if trajectory.length == STEPS_PER_TRAJECTORY:
                    last_state = trajectory.get_last()
                    trajectory.finish()
                    queue.put(trajectory)
                    break

                current_state = next_state
                steps += 1
    env.close()
    mean_rewards[p_id] = cumulative_reward / num_episodes
    terminated[p_id] = 1
