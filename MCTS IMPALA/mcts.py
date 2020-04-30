import numpy as np
import gym
from copy import deepcopy


class Node:
    """
    Attributes:
        state: the state value
        n_visits: number of visits made to the state
        children: nodes containing the next states
    """

    def __init__(self, state=None, value=0, parent=None):
        self.state = state
        self.value = value
        self.n_visits = 0
        self.parent = parent
        self.children = dict()

    def add_child(self, action, next_state):
        self.children[action] = next_state

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_rollouts = 10
        self.rollout_depth = 20

    """
    Select the node with the maximum upper confidence bound
    """

    def __selection(self, node):
        next_action = max([(action, self.__upper_confidence_bound(child)) for action, child in node.children.items()],
                          key=lambda el: el[1])[0]
        return next_action, node.children[next_action]

    def __upper_confidence_bound(self, node):
        return np.inf if node.n_visits == 0 else node.value + 2 * np.sqrt(node.parent.n_visits / node.n_visits)

    """
    Expand the current leaf node. Choose one of the children at random and estimate value using simulations.
    """

    def __expansion(self, current_state_node, env):
        unexplored_children = {action: Node(parent=current_state_node) for action in range(self.n_actions)}
        current_state_node.children = unexplored_children

        random_action = np.random.choice(self.n_actions)
        next_state, reward, done, _ = env.step(random_action)
        value_of_next_state = self.__rollout(env)

        next_state_node = current_state_node.children.get(random_action)
        next_state_node.state = next_state
        next_state_node.value = value_of_next_state

        return random_action, next_state_node

    """
    Performs multiple rollouts from the current state to get value function estimate
    """

    def __rollout(self, env):
        return np.mean([self.__single_rollout(env) for idx in range(self.n_rollouts)])

    def __single_rollout(self, env):
        temp_env = deepcopy(env)
        done = False
        total_reward = 0
        while not done:
            random_action = np.random.choice(temp_env.action_space.n)
            _, reward, done, _ = temp_env.step(random_action)
            total_reward += reward
        temp_env.close()
        return total_reward

    """
    Learn the traversal policy
    """

    def chose_action(self, state):
        root = Node(state)
        for i in range(self.rollout_depth):
            temp_env = deepcopy(self.env)
            current_state_node = root
            while not current_state_node.is_leaf():
                next_action, next_state_node = self.__selection(current_state_node)
                temp_env.step(next_action)
                current_state_node = next_state_node
            action_chosen, next_state_node = self.__expansion(current_state_node, temp_env)
            self.backprop(current_state_node, next_state_node.value)
            temp_env.close()

        next_action, _ = self.__selection(root)
        return next_action

    """
    Backpropagate the estimated values and increment the visit counts
    """

    def backprop(self, node, value):
        while node:
            node.value += value
            node.n_visits += 1
            node = node.parent


def main():
    env = gym.make('CartPole-v1')
    done = False
    total_reward = 0
    state = env.reset()
    while not done:
        next_action = MCTS(deepcopy(env)).chose_action(state)
        state, reward, done, _ = env.step(next_action)
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    scores = [main() for i in range(10)]
    print(scores)
    print(np.mean(scores))
