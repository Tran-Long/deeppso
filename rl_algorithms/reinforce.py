import torch

from envs import BaseEnv

from .base import BaseRLAlgorithm


class REINFORCE(BaseRLAlgorithm):
    """
    REINFORCE (Williams, 1992) for the PSO meta-controller.

    Per-step, per-particle reward:
        r_t[b, p] = -cost_t[b, p]

        The negative tour length decoded from particle p's position at step t.
        This varies across the P particles in each swarm (different positions
        yield different tours and different costs), making the particle-mean
        baseline a genuine variance-reduction tool rather than cancelling the
        entire signal.

    Discounted return (backward induction):
        G_t[b, p] = Σ_{k=t}^{T-1} γ^{k-t} · r_k[b, p]

        Unlike pl_raw.py which uses the immediate reward r_t as the gradient
        weight, REINFORCE uses G_t — the agent gets credit for actions that
        lead to good outcomes throughout the remaining trajectory, not just the
        next step.  γ < 1 discounts distant future rewards and reduces variance.

    Particle-mean baseline (per step, per problem):
        b_t[b] = mean_p( G_t[b, p] )

        Unbiased variance reduction.  Valid here because G_t[b, p] differs
        across particles (costs vary per position).  The advantage measures
        how much better particle p's trajectory was relative to its swarm.

    Advantage normalisation (per step, per problem):
        Â_t[b, p] = (G_t[b, p] - b_t[b]) / std_p(G_t[b, p])

        Normalises gradient scale independently per problem, handling
        different city-count instances in the same batch naturally.

    Loss:
        L = -mean_{t,b,p}[ Â_t[b,p] · log π(a_t[b,p] | s_t[b,p]) ] - β·H[π]

    Implementation notes
    --------------------
    - `problem_embedding` is computed once and shared across all T steps;
      a single backward after the loop gives the GNN gradients from every
      timestep simultaneously.
    - Returns are detached before use as weights; only log_probs carries the
      gradient path.
    - γ=1.0 (undiscounted) works well for short horizons (T≤50). Use a
      smaller value to sharpen credit assignment on longer rollouts.
    """

    def __init__(
        self,
        *args,
        norm_rewards: bool = True,
        gamma: float = 1.0,
        entropy_coef: float = 0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.norm_rewards = norm_rewards

    def training_step(self, env: BaseEnv, idx):
        observations, _ = env.reset()
        opt = self.optimizers()
        opt.zero_grad()
        T = self.pso_iterations_train
        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations.
        problem_embedding = self.agent.get_problem_embedding(env.problem)

        # ------------------------------------------------------------------ #
        # Phase 1: roll out the full PSO episode, collecting the trajectory.  #
        # ------------------------------------------------------------------ #
        # all_log_probs[t] : (B, P) — kept in the live computation graph
        # all_rewards[t]   : (B, P) — per-particle tour costs, detached
        # all_entropies[t] : (B, P) — kept in the live computation graph
        all_log_probs = torch.zeros(
            (T, env.batch_size), device=self.device
        )  # (T, B, P) → (T, B) after mean over particles
        all_rewards = torch.zeros((T, env.batch_size), device=self.device)  # (T, B)
        all_entropies = torch.zeros((T, env.batch_size), device=self.device)  # (T, B)
        all_dones = torch.ones(
            (T, env.batch_size), dtype=torch.bool, device=self.device
        )  # (T, B) assume all done until we see otherwise

        is_env_dones = torch.zeros(
            (env.batch_size,), dtype=torch.bool, device=self.device
        )  # (B,)
        done = torch.zeros(
            (env.batch_size,), dtype=torch.bool, device=self.device
        )  # (B,)
        for _iter in range(self.pso_iterations_train):
            is_env_dones = (
                is_env_dones | done
            )  # Done is taken from the previous step. Once a problem is done, it stays done.
            if is_env_dones.all():
                break

            all_dones[_iter] = is_env_dones
            wc1c2, log_probs, entropy = self.agent.get_action(
                (*observations, problem_embedding)
            )
            observations, reward, done, _, _ = env.step_train(
                wc1c2,
                using_random=self.pso_using_random,
            )
            all_log_probs[_iter] = log_probs.mean(dim=-1)
            all_rewards[_iter] = (
                reward if len(reward.shape) == 1 else reward.mean(dim=-1)
            )  # Handle both per-particle and already-meaned rewards.
            all_entropies[_iter] = entropy.mean(dim=-1)

        # ------------------------------------------------------------------ #
        # Phase 2: discounted returns via backward induction.                 #
        # ------------------------------------------------------------------ #
        returns = torch.zeros_like(all_rewards)  # (T, B)
        future_return = torch.zeros((env.batch_size,), device=self.device)  # (B,)
        for t in reversed(range(T)):
            reward_t = all_rewards[t]  # (B,)
            done_t = all_dones[t]  # (B,)
            future_return = (reward_t + self.gamma * future_return) * (~done_t)  # (B,)
            returns[t] = future_return

        # ------------------------------------------------------------------ #
        # Phase 3: particle-mean baseline + normalised advantage.             #
        # ------------------------------------------------------------------ #

        # Normalise per (step, problem) for scale-invariance across problem
        # sizes (20-city vs 100-city instances in the same batch).
        if self.norm_rewards:
            returns = (returns - returns.mean(dim=-1, keepdim=True)) / (
                returns.std(dim=-1, keepdim=True).clamp(min=1e-8)
            )  # Normalise returns per step for stability.

        reinforce_loss = -(
            returns * all_log_probs
        ).mean()  # REINFORCE policy gradient loss.
        entropy_loss = (
            -self.entropy_coef * all_entropies.mean()
        )  # Entropy regularisation loss.
        total_loss = reinforce_loss + entropy_loss

        # ------------------------------------------------------------------ #
        # Phase 4: single backward + optimizer step.                          #
        # ------------------------------------------------------------------ #
        self.manual_backward(total_loss)
        self.clip_gradients(
            opt, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm"
        )
        opt.step()
        opt.zero_grad()

        self.log("train_loss", total_loss.detach())
        self.log("train_gbest", env.val_gbest.mean())
