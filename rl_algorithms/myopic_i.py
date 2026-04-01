import torch

from envs import BaseEnv

from .base import BaseRLAlgorithm


class MyopicI(BaseRLAlgorithm):
    # Single-step episode: only the immediate reward is used for the policy gradient, no credit assignment across PSO iterations.
    # REINFORCE
    def training_step(self, env: BaseEnv, idx):
        observations, _ = env.reset()
        opt = self.optimizers()
        for _iter in range(self.pso_iterations_train):
            # Temperature annealing: high early (explore) → low late (exploit)
            progress = _iter / max(self.pso_iterations_train - 1, 1)
            temperature = 2.0 * (1.0 - progress) + 0.5 * progress

            opt.zero_grad()
            problem_embedding = self.agent.get_problem_embedding(env.problem)
            wc1c2, log_probs, entropy = self.agent.get_action(
                (*observations, problem_embedding)
            )
            observations, reward, _, _, _ = env.step_train(
                wc1c2,
                temperature=temperature,
                using_random=self.pso_using_random,
            )
            # REINFORCE with per-problem baseline
            # Shapes: reward (B, P) or (B,), log_probs (B, P), entropy (B, P)
            #   - reward[b,p] = negative stochastic tour cost of particle p in problem b
            #   - log_probs[b,p] = log π(action | state) for particle p, summed over 3 params, meaned over D edges
            #   - baseline[b] = mean reward across particles in problem b
            #   - advantage[b,p] = how much better/worse particle p did vs. swarm average
            baseline = reward.mean(dim=-1, keepdim=True)  # (B, 1)
            advantage = reward - baseline  # (B, P) or (B,)
            if len(reward.shape) == 2:
                reinforce_loss = -(
                    advantage * log_probs
                ).mean()  # scalar: mean over B×P
            elif len(reward.shape) == 1:
                reinforce_loss = -(
                    advantage * log_probs.mean(dim=-1)
                ).mean()  # scalar: mean over B

            loss = reinforce_loss - 0.01 * entropy.mean()
            # Single backward + optimizer step after all PSO iterations.
            # tsp_embedding's graph is traversed exactly once here.
            self.manual_backward(loss)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.max_grad_norm,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad()

            self.log("train_loss", loss.detach())
