from functools import reduce

import torch

from envs import BaseEnv, BaseProblem

from .base import BaseRLAlgorithm


class PPO(BaseRLAlgorithm):
    def __init__(
        self,
        *args,
        use_gae=True,
        clip_coef=0.2,
        clip_vloss=False,
        entropy_coef=0.01,
        vf_coef=0.5,
        gae_lambda=0.95,
        gamma=0.99,
        num_updates=4,
        mini_batch_size=64,
        norm_adv=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert (
            getattr(self.agent, "critic", None) is not None
        ), "PPO requires the agent to have a critic for value estimation."
        assert (
            self.agent.get_action_and_value is not None
        ), "PPO agent must implement get_action_and_value method that returns both action and value estimates."
        assert (
            self.agent.get_value is not None
        ), "PPO agent must implement get_value method that returns value estimates."
        self.use_gae = use_gae
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.num_updates = num_updates
        self.mini_batch_size = mini_batch_size
        self.norm_adv = norm_adv

    def training_step(self, env: BaseEnv, idx):
        assert (
            env.auto_reset
        ), "Vectorized environments must have auto-reset enabled for PPO training."
        observations, _ = env.reset()
        opt = self.optimizers()
        T = self.pso_iterations_train
        B = env.batch_size
        P = env.n_particles
        D = env.dim

        all_obs_pop = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_vel = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_pbest = torch.zeros((T, B, P, D), device=self.device)  # (T, B, P, D)
        all_obs_gbest = torch.zeros((T, B, D), device=self.device)  # (T, B, D)
        all_obs_problems: list[BaseProblem] = []
        problem_cls = type(env.problem)

        all_actions = torch.zeros(
            (T, B, P, D, 3), device=self.device
        )  # (T, B, P, D, 3)
        all_log_probs = torch.zeros((T, B), device=self.device)  # (T, B)
        all_rewards = torch.zeros((T, B), device=self.device)  # (T, B)
        all_entropies = torch.zeros((T, B), device=self.device)  # (T, B)
        all_dones = torch.zeros((T, B), dtype=torch.bool, device=self.device)  # (T, B)
        all_values = torch.zeros((T, B), device=self.device)  # (T, B)

        done = torch.zeros(
            (env.batch_size,), dtype=torch.bool, device=self.device
        )  # (B,)
        for _iter in range(T):
            population, velocity, pbest, gbest, problem = observations

            # For observations
            all_obs_pop[_iter] = population
            all_obs_vel[_iter] = velocity
            all_obs_pbest[_iter] = pbest
            all_obs_gbest[_iter] = gbest
            all_obs_problems.append(problem)

            all_dones[_iter] = done

            with torch.no_grad():
                (wc1c2, log_probs, entropy), value = self.agent.get_action_and_value(
                    (*observations, None)
                )

            observations, reward, done, _, _ = env.step_train(
                wc1c2,
                using_random=self.pso_using_random,
            )
            all_actions[_iter] = wc1c2
            all_log_probs[_iter] = log_probs.mean(dim=-1)
            all_rewards[_iter] = (
                reward if len(reward.shape) == 1 else reward.mean(dim=-1)
            )  # Handle both per-particle and already-meaned rewards.
            all_entropies[_iter] = entropy.mean(dim=-1)
            all_values[_iter] = value.squeeze(-1)  # Assuming critic returns (B, 1)

        with torch.no_grad():
            terminate_values = self.agent.get_value((*observations, None)).squeeze(
                -1
            )  # (B,)
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
                delta = (
                    all_rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - all_values[t]
                )
                advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )
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
                returns[t] = (
                    all_rewards[t] + self.gamma * next_return * next_non_terminal
                )
            advantages = returns - all_values

        # Flatten the (T, B, ...) tensors to (T*B, ...)
        flat_obs_pop = all_obs_pop.view(T * B, P, D)
        flat_obs_vel = all_obs_vel.view(T * B, P, D)
        flat_obs_pbest = all_obs_pbest.view(T * B, P, D)
        flat_obs_gbest = all_obs_gbest.view(T * B, D)
        flat_obs_problems = reduce(
            lambda x, y: x + y,
            [problem_cls.unbatch_instances(prob) for prob in all_obs_problems],
        )  # List of length T*B
        flat_actions = all_actions.view(T * B, P, D, 3)
        flat_log_probs = all_log_probs.view(T * B)
        flat_advantages = advantages.view(T * B)
        flat_returns = returns.view(T * B)
        flat_values = all_values.view(T * B)

        # Optimize policy for K epochs:
        for _ in range(self.num_updates):
            # Generate mini-batches of indices
            indices = torch.randperm(T * B, device=self.device)
            for start in range(0, T * B, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_obs_pop = flat_obs_pop[mb_indices]
                mb_obs_vel = flat_obs_vel[mb_indices]
                mb_obs_pbest = flat_obs_pbest[mb_indices]
                mb_obs_gbest = flat_obs_gbest[mb_indices]
                mb_obs_problems = problem_cls.batch_instances(
                    *[flat_obs_problems[i] for i in mb_indices]
                )
                mb_actions = flat_actions[mb_indices]
                mb_log_probs = flat_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]
                mb_values = flat_values[mb_indices]

                (new_wc1c2, new_log_probs, new_entropy), new_value = (
                    self.agent.get_action_and_value(
                        (
                            mb_obs_pop,
                            mb_obs_vel,
                            mb_obs_pbest,
                            mb_obs_gbest,
                            mb_obs_problems.to(self.device),
                            None,
                        )
                    )
                )
                new_log_probs_mean = new_log_probs.mean(dim=-1)

                ratio = (new_log_probs_mean - mb_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss_unclipped = (
                    (new_value.squeeze(-1) - mb_returns) ** 2
                ).mean()
                if self.clip_vloss:
                    value_clipped = mb_values + torch.clamp(
                        new_value.squeeze(-1) - mb_values,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    value_loss_clipped = (value_clipped - mb_returns) ** 2
                    value_loss = torch.max(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                else:
                    value_loss = value_loss_unclipped

                entropy_loss = -new_entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )
                opt.step()
