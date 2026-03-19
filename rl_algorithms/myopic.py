import torch

from envs import BaseEnvPSOBatchProblem

from .base import BaseRLAlgorithm


class Myopic(BaseRLAlgorithm):
    def training_step(self, env: BaseEnvPSOBatchProblem, idx):
        observations, _ = env.reset()
        opt = self.optimizers()
        opt.zero_grad()
        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations

        problem_embedding = self.agent.get_problem_embedding(env.problem)

        # Accumulate losses as a live computation graph sum — do NOT call
        # manual_backward inside the loop.  A single backward at the end
        # traverses the tsp_embedding graph exactly once, so there is no
        # "backward through freed graph" error AND the GNN gets its gradients.
        total_loss = torch.zeros(1, device=self.device)
        for _iter in range(self.pso_iterations_train):
            # Temperature annealing: high early (explore) → low late (exploit)
            progress = _iter / max(self.pso_iterations_train - 1, 1)
            temperature = 2.0 * (1.0 - progress) + 0.5 * progress

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
                reinforce_loss = -(advantage * log_probs).mean()  # scalar: mean over B×P
            elif len(reward.shape) == 1:
                reinforce_loss = -(advantage * log_probs.mean(dim=-1)).mean()  # scalar: mean over B

            loss = reinforce_loss - 0.01 * entropy.mean()
            # Add to the live graph — backward is deferred until after the loop.
            total_loss = total_loss + loss

        # Normalize by number of iterations so gradient scale is independent of T.
        total_loss = total_loss / self.pso_iterations_train
        # Single backward + optimizer step after all PSO iterations.
        # tsp_embedding's graph is traversed exactly once here.
        self.manual_backward(total_loss)
        self.clip_gradients(opt, gradient_clip_val=1.7, gradient_clip_algorithm="norm")
        opt.step()
        opt.zero_grad()

        avg_loss = total_loss.detach()
        self.log("train_loss", avg_loss)
        self.log(
            "train_gbest",
            env.val_gbest.mean(),
        )
