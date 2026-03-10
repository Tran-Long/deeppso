import torch

from envs import BaseEnvPSOBatchProblem

from .base import BaseRLAlgorithm


class REINFORCE(BaseRLAlgorithm):
    """
    True REINFORCE (Williams, 1992) for the PSO meta-controller.

    Each PSO episode of length T is treated as a finite-horizon trajectory:
        s_0, a_0, r_0, s_1, a_1, r_1, ..., s_{T-1}, a_{T-1}, r_{T-1}

    Reward definition (shape B × P, from TSPEnvVectorEdgeBatch.step):
        r_t[b, p] = -val_gbest[b]   (identical for all P particles)

        Dense and non-zero at every step. Since val_gbest is monotone
        non-increasing, the discounted return G_t rewards trajectories that
        reach a low-cost solution *early* — directly incentivising faster
        convergence, which is the stated goal.

    The gradient weight for log π(a_t | s_t) is the **discounted return**:
        G_t = Σ_{k=t}^{T-1}  γ^{k-t} · r_k

    A per-step **batch-mean** baseline is subtracted for variance reduction
    (does not bias the gradient):
        b_t = mean_b( G_t[b, p] )   for each particle p

    NOTE: a particle-mean baseline MUST NOT be used here. Because r_t[b,p]
    is identical for all p, mean_p(G_t[b,p]) == G_t[b,p], which produces
    zero advantage everywhere and kills all gradients.

    Loss (averaged over B × P × T):
        L = -E[ (G_t - b_t) · log π(a_t | s_t) ] - β · H[π]

    Implementation notes
    --------------------
    - `problem_embedding` is computed once and shared across all T steps.
      Accumulating all per-step losses into `total_loss` before calling a
      single `manual_backward` ensures the GNN receives gradients from every
      timestep in one backward pass — identical pattern to `pl_raw.py`.
    - Returns are `detach()`-ed before use as weights so they never flow
      gradients themselves; only `log_probs` carries the gradient path.
    - `gamma=1.0` (undiscounted) is the default for short PSO horizons (T≤50).
      Pass a smaller value to trade off credit-assignment range vs. variance.
    """

    def __init__(self, *args, gamma: float = 1.0, entropy_coef: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def training_step(self, env: BaseEnvPSOBatchProblem, idx):
        observations, _ = env.reset()
        opt = self.optimizers()
        opt.zero_grad()

        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations.
        problem_embedding = self.agent.get_problem_embedding(env.problem)

        # ------------------------------------------------------------------ #
        # Phase 1: roll out the full PSO episode, collecting the trajectory.  #
        # ------------------------------------------------------------------ #
        # all_log_probs[t] : (B, P)  — kept in the live computation graph
        # all_rewards[t]   : (B, P)  — detached costs, no grad needed
        # all_entropies[t] : (B, P)  — kept in the live computation graph
        all_log_probs = []
        all_rewards = []
        all_entropies = []

        for _iter in range(self.pso_iterations_train):
            wc1c2, log_probs, entropy = self.agent.get_action(
                (*observations, problem_embedding)
            )
            observations, reward, _, _, _ = env.step_train(
                wc1c2,
                using_random=self.pso_using_random,
            )
            all_log_probs.append(log_probs)  # keep grad
            all_rewards.append(reward.detach())  # purely numerical
            all_entropies.append(entropy)  # keep grad

        # ------------------------------------------------------------------ #
        # Phase 2: compute discounted returns G_t via backward induction.     #
        # ------------------------------------------------------------------ #
        T = self.pso_iterations_train
        # rewards_tensor: (T, B, P)
        rewards_tensor = torch.stack(all_rewards, dim=0)

        returns = torch.zeros_like(rewards_tensor)  # (T, B, P)
        returns[-1] = rewards_tensor[-1]
        for t in range(T - 2, -1, -1):
            returns[t] = rewards_tensor[t] + self.gamma * returns[t + 1]

        # ------------------------------------------------------------------ #
        # Phase 3: build the total loss using returns as (detached) weights.  #
        # ------------------------------------------------------------------ #
        # Baseline: mean return across problem instances for each (step, particle).
        #   baseline: (T, 1, P) → broadcasts over B
        # Must use batch-mean (dim=1), NOT particle-mean (dim=-1):
        # reward = -val_gbest is broadcast identically to all P particles, so
        # particle-mean == the value itself → zero advantage → zero gradient.
        baseline = returns.mean(dim=1, keepdim=True)
        advantage = returns - baseline  # (T, B, P)

        # Normalise advantages over the batch dimension (B) per step.
        # dim=1 (B), not dim=-1 (P): since reward is identical across P,
        # std over P == 0 and would only amplify floating-point noise.
        std = advantage.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantage = advantage / std  # (T, B, P)

        # log_probs_tensor: (T, B, P) — still in the computation graph
        log_probs_tensor = torch.stack(all_log_probs, dim=0)
        entropy_tensor = torch.stack(all_entropies, dim=0)

        # Weight log-probs by detached advantages.
        # Single backward: PyTorch accumulates gradients through every
        # log_probs_tensor[t] back to problem_embedding in one pass.
        reinforce_loss = -(advantage.detach() * log_probs_tensor).mean()
        entropy_loss = -self.entropy_coef * entropy_tensor.mean()
        total_loss = reinforce_loss + entropy_loss

        # ------------------------------------------------------------------ #
        # Phase 4: single backward + optimizer step.                          #
        # ------------------------------------------------------------------ #
        self.manual_backward(total_loss)
        self.clip_gradients(opt, gradient_clip_val=1.7, gradient_clip_algorithm="norm")
        opt.step()
        opt.zero_grad()

        self.log("train_loss", total_loss.detach())
        self.log("train_gbest", env.val_gbest.mean())
