import torch as T
import torch.nn.functional as F
import numpy as np

# Need to get discounts from learner
def vtrace_target(actor_action_distribution, learner_action_distribution, actions, rewards, values, discounts, bootstrap_value,
                  clip_rho_threshold = 1.0, clip_rho_pg_threshold = 1.0, clip_c_threshold = 1.0):
    actor_action_log_probability = log_probabilities(actor_action_distribution, actions)
    learner_action_log_probability = log_probabilities(learner_action_distribution, actions)

    log_likelihood_ratio = learner_action_log_probability - actor_action_log_probability

    clip_rho_threshold = T.tensor(clip_rho_threshold, dtype=T.float32)
    clip_rho_pg_threshold = T.tensor(clip_rho_pg_threshold, dtype=T.float32)
    clip_c_threshold = T.tensor(clip_c_threshold, dtype=T.float32)

    with T.no_grad():
        likelihood_ratio = T.exp(log_likelihood_ratio)
        clipped_rho = T.min(clip_rho_threshold, likelihood_ratio)
        clipped_c = T.min(clip_c_threshold, likelihood_ratio)

        values_t_plus_1 = T.cat([values[1:], bootstrap_value.view(1, -1)])

        # V-trace Targets
        # delta_t*V
        deltas = clipped_rho * (rewards + discounts * values_t_plus_1 - values)
        trajectory_length = actions.shape[0]
        vs = [np.sum([values[t], v_trace_target_at_time_t(t, trajectory_length, discounts, clipped_c, deltas)]) for t in range(trajectory_length)]
        vs = T.stack(vs, dim = 0)

        vs_t_plus_1 = T.cat([vs[1:], bootstrap_value.view(1, -1)])
        # Policy Gradient Advantage
        clipped_pg_rho = T.min(clip_rho_pg_threshold, likelihood_ratio)
        advantage = rewards + discounts * vs_t_plus_1 - values
        policy_gradient_advantage = clipped_pg_rho * advantage

        return vs, policy_gradient_advantage


def v_trace_target_at_time_t(t, trajectory_length, discounts, clipped_c, deltas):
    intermediate_values = [T.prod(discounts[t:timestep]) *
                           T.prod(clipped_c[t:timestep]) *
                           deltas[timestep]
     for timestep in range(t, trajectory_length)]
    return T.sum(T.stack(intermediate_values), dim = 0)


# Target needs to be of type long: https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
def log_probabilities(action_probabilities, action):
    return -F.cross_entropy(action_probabilities, action.to(T.long), reduction = "none")



