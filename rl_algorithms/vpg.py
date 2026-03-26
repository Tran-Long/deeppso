from functools import reduce

import torch
import torch.nn.functional as F

from envs import BaseEnvPSOBatchProblem, BaseProblem

from .base import BaseRLAlgorithm


class VPG(BaseRLAlgorithm):
    def __init__(self, 
                 *args, 
                 gamma=0.99, 
                 use_gae=True, 
                 entropy_coef=0.01, 
                 vf_coef=0.5,
                 gae_lambda=0.95, 
                 norm_adv=True, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert getattr(self.agent, "critic", None) is not None, "VPG requires the agent to have a critic for value estimation."
        assert self.agent.get_action_and_value is not None, "VPG agent must implement get_action_and_value method that returns both action and value estimates."
        assert self.agent.get_value is not None, "VPG agent must implement get_value method that returns value estimates."
        self.gamma = gamma
        self.norm_adv = norm_adv
        self.use_gae = use_gae
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.gae_lambda = gae_lambda

    def training_step(self, env: BaseEnvPSOBatchProblem, idx):
        assert env.auto_reset, "Vectorized environments must have auto-reset enabled for VPG training."
        observations, _ = env.reset()
        opt = self.optimizers()
        T = self.pso_iterations_train; B = env.batch_size; P = env.n_particles; D = env.dim
        # assert B == 1, "VPG implementation currently only supports batch size of 1 for simplicity. Got batch size: {}".format(B)

        all_obs_pop = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_vel = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_pbest = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_gbest = torch.zeros((T, B, D), device=self.device)  # (T, B, D)
        all_obs_problems: list[BaseProblem] = [] 
        problem_cls = type(env.problem)

        all_actions = torch.zeros((T, B, P, D, 3), device=self.device)  # (T, B, P, D, 3)
        all_log_probs = torch.zeros((T, B, P), device=self.device)  # (T, B, P)
        all_rewards = torch.zeros((T, B, P), device=self.device)  # (T, B, P)
        all_entropies = torch.zeros((T, B, P), device=self.device)  # (T, B, P)
        all_dones = torch.zeros((T, B, P), dtype=torch.bool, device=self.device)  # (T, B, P)
        all_values = torch.zeros((T, B, P), device=self.device)  # (T, B, P)

        done = torch.zeros((B, P), dtype=torch.bool, device=self.device)  # (B, P)
        
        for t in range(T):
            population, velocity, pbest, gbest, problem = observations

            # Store observations
            all_obs_pop[t] = population
            all_obs_vel[t] = velocity
            all_obs_pbest[t] = pbest
            all_obs_gbest[t] = gbest
            all_obs_problems.append(problem)  # Store a copy of the problem instance
            all_dones[t] = done

            (wc1c2, log_probs, entropy), value = self.agent.get_action_and_value((*observations, None))
            
            observations, rewards, done, _, _ = env.step_train(
                wc1c2=wc1c2,
                using_random=self.pso_using_random
            )
            done = done.unsqueeze(-1).expand_as(rewards)  # (B, P)
            assert rewards.shape == (B, P), f"Expected rewards shape to be (B, P) but got {rewards.shape}"

            # Store actions, log probs, rewards, entropies, values
            all_actions[t] = wc1c2
            all_log_probs[t] = log_probs
            all_rewards[t] = rewards
            all_entropies[t] = entropy
            all_values[t] = value
            # if done.all():
            #     # Only here because of the batchsize = 1 assumption. 
            #     # In a more general implementation
            #     break

        with torch.no_grad():
            terminate_values = self.agent.get_value((*observations, None))
        if self.use_gae:
            advantages = torch.zeros_like(all_rewards)
            last_gae_lambda = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - done.float()
                    next_values = terminate_values
                else:
                    next_non_terminal = 1.0 - all_dones[t + 1].float()
                    next_values = all_values[t + 1]
                delta = all_rewards[t] + self.gamma * next_values * next_non_terminal - all_values[t]
                advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            returns = advantages + all_values
        else:
            returns = torch.zeros_like(all_rewards)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - done.float()
                    next_return = terminate_values
                else:
                    next_non_terminal = 1.0 - all_dones[t + 1].float()
                    next_return = returns[t + 1]
                returns[t] = all_rewards[t] + self.gamma * next_return * next_non_terminal
            advantages = returns - all_values

        advantages = advantages.detach()
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(all_log_probs * advantages).mean()
        value_loss = (returns - all_values).pow(2).mean()
        entropy_loss = -all_entropies.mean()

        loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
