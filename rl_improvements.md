# RL Training Improvements — VPG with GAE & PPO

## Current Setup

The PSO loop is an RL episode where:
- **State** $s_t$: `(pos, vel, pbest, gbest, tsp_embedding)` at iteration $t$
- **Action** $a_t$: sampled `wc1c2` from `Normal(mu, sigma)` — shape `(n_particles, dim, 3)`
- **Reward** $r_t$: `-cost_stochastic` (negative tour cost from stochastic decode)
- **Policy** $\pi_\theta(a_t | s_t)$: the `HyperparamTSPModel` network

Currently using bare REINFORCE with a mean-reward baseline:
```python
reward = -costs.detach()
baseline = reward.mean()
loss = -((reward - baseline) * log_prob_per_particle).mean()
```

This has **high variance** and **no temporal credit assignment** — every iteration gets the same single-step reward with no notion of how current actions affect future returns.

---

## What To Build

### Option A: Vanilla Policy Gradient with Learned Value Function & GAE

**Core idea:** Learn a value function $V_\phi(s_t)$ that predicts expected return from state $s_t$. Use it to compute Generalized Advantage Estimation (GAE), which provides a low-variance, bias-controllable advantage signal.

#### 1. Value Network (`ValueHead`)

**Where:** New class in `nets/tsp_vector.py` (or a new `nets/value_head.py`).

**Architecture:** Reuse the same encoder path as the policy — the `particle_ctx` from cross-attention already encodes the full state. Add a small value head on top:

```
particle_ctx: (n_particles, emb_dim)   ← from the SAME cross-attention as the policy
                     │
              MLP(emb_dim → emb_dim → 1)
                     │
              V(s_t): (n_particles,)        ← predicted return per particle
                     │
            mean over particles
                     │
              V(s_t): scalar              ← single state value (optional design choice)
```

**Design choice — per-particle vs. global value:**
- **Per-particle** `V(s_t, p)`: each particle gets its own value estimate. Advantage is per-particle: $A_t^p = r_t^p - V(s_t, p) + \gamma V(s_{t+1}, p)$. More precise but noisier for small swarms.
- **Global** `V(s_t) = mean_p V(s_t, p)`: single baseline for all particles. Simpler. Works well because all particles share the same PSO dynamics.

**Recommendation:** Start with **per-particle** values. Each particle sees a different reward (different cost from stochastic decode), so per-particle baselines reduce variance more.

**Weight sharing:** The value head shares the TSP encoder, SwarmEncoder, and cross-attention with the policy. Only the final MLP head is separate. This is standard in actor-critic architectures and keeps the model lightweight.

```python
class HyperparamTSPModel(nn.Module):
    def __init__(self, ...):
        ...
        # existing policy heads
        self.edge_head_mu = MLP(...)
        self.edge_head_sigma = MLP(...)
        # NEW: value head — operates on particle_ctx (global per-particle embedding)
        self.value_head = MLP(
            units_list=[emb_dim, emb_dim, 1],
            act_fn=act_fn,
        )

    def forward(self, ...) -> tuple:
        ...
        particle_ctx = ...  # (n_particles, emb_dim) — already computed
        # Policy outputs (unchanged)
        wc1c2_mu, wc1c2_sigma = ...  # (n_particles, dim, 3) each
        # Value output
        values = self.value_head(particle_ctx).squeeze(-1)  # (n_particles,)
        return wc1c2_mu, wc1c2_sigma, values
```

#### 2. GAE Computation

After rolling out the full PSO trajectory (all `pso_iterations_train` steps), compute GAE **post-hoc**:

```python
# Collected during the loop:
rewards    = [r_0, r_1, ..., r_{T-1}]   # each (n_particles,)
values     = [V_0, V_1, ..., V_{T-1}]   # each (n_particles,)
log_probs  = [lp_0, lp_1, ..., lp_{T-1}] # each (n_particles,)

# After the loop, append bootstrap value for terminal state:
V_T = 0  # episode ends after PSO loop — no future reward

# GAE (vectorised over particles):
gamma = 0.99
lam = 0.95
advantages = []
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * (values[t+1] if t < T-1 else 0) - values[t]
    gae = delta + gamma * lam * gae
    advantages.insert(0, gae)

returns = [adv + val for adv, val in zip(advantages, values)]
```

$\gamma$ and $\lambda$ are hyperparameters:
- $\gamma$: discount factor. For short episodes (T=32–64), use **0.99**.
- $\lambda$: GAE bias-variance trade-off. $\lambda=1$ → high variance (like REINFORCE), $\lambda=0$ → high bias (1-step TD). Use **0.95**.

#### 3. Loss Function

```python
# Policy loss (same structure as REINFORCE but with GAE advantages)
advantages = torch.stack(advantages).detach()       # (T, n_particles)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalise
log_probs = torch.stack(log_probs)                   # (T, n_particles)
policy_loss = -(advantages * log_probs).mean()

# Value loss
returns = torch.stack(returns).detach()              # (T, n_particles)
values = torch.stack(values)                         # (T, n_particles)
value_loss = F.mse_loss(values, returns)

# Entropy bonus (unchanged)
entropy = ...

# Total
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

**Value loss coefficient 0.5** is standard — prevents the value head from dominating gradients since MSE can be large early in training.

#### 4. Training Loop Change

The loop structure changes from "accumulate loss inside loop" to "collect trajectory, then compute loss":

```python
# Phase 1: Rollout (collect trajectory)
trajectory = {"rewards": [], "values": [], "log_probs": [], "entropies": []}
for _iter in range(T):
    mu, sig, value = self.net(...)
    d = Normal(mu, sig)
    action = d.sample()
    log_prob = d.log_prob(action).sum(-1).mean(-1)  # (n_particles,)
    ...  # PSO step + decode
    reward = -costs.detach()
    trajectory["rewards"].append(reward)
    trajectory["values"].append(value)
    trajectory["log_probs"].append(log_prob)
    trajectory["entropies"].append(d.entropy().sum(-1).mean())

# Phase 2: Compute GAE
advantages, returns = compute_gae(trajectory["rewards"], trajectory["values"], gamma=0.99, lam=0.95)

# Phase 3: Compute losses
policy_loss = -(torch.stack(advantages).detach() * torch.stack(trajectory["log_probs"])).mean()
value_loss = F.mse_loss(torch.stack(trajectory["values"]), torch.stack(returns).detach())
entropy_bonus = torch.stack(trajectory["entropies"]).mean()
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

self.manual_backward(total_loss)
```

#### 5. Optimizer

Use a **single** Adam optimiser for both policy and value (they share the encoder). The value head's separate parameters are included via `self.net.parameters()` which already covers everything.

---

### Option B: Proximal Policy Optimization (PPO)

**Core idea:** After collecting a trajectory (one PSO rollout), perform **multiple gradient steps** on the same data using a clipped surrogate objective. This extracts more learning signal from each episode.

**PPO is on-policy.** There is no replay buffer. The workflow each training step is:
1. Roll out the **current** policy to collect one trajectory
2. Compute GAE advantages from that trajectory
3. Do K gradient updates on **that same trajectory** (clipped surrogate prevents the policy from moving too far)
4. **Discard** the trajectory entirely — it is never reused

The trajectory is held in memory only for the duration of the K update epochs, then freed. This is fundamentally different from off-policy methods (SAC, TD3) that maintain a persistent replay buffer of old experiences.

PPO requires all the same components as VPG+GAE (value network, GAE computation), plus:

#### 1. Ephemeral Trajectory Data (Not a Replay Buffer)

During rollout, additionally store the **old log-probs** (under the policy that generated the trajectory). These are used to compute importanceweighting ratios during the K re-evaluation epochs:

```python
trajectory["old_log_probs"] = []  # log π_old(a_t | s_t)
```

These are detached — they don't participate in the gradient.

#### 2. PPO Clipped Surrogate Loss

After computing GAE advantages, perform `K` mini-epochs of optimisation on the trajectory:

```python
eps_clip = 0.2  # standard PPO clip range
K_epochs = 4    # number of passes over the trajectory

old_log_probs = torch.stack(trajectory["old_log_probs"]).detach()  # (T, n_particles)
advantages = torch.stack(advantages).detach()
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns = torch.stack(returns).detach()

for _ in range(K_epochs):
    # Re-evaluate the CURRENT policy on the stored states
    new_log_probs = []
    new_values = []
    new_entropies = []
    for t in range(T):
        mu, sig, value = self.net(states[t]...)  # re-forward the stored states
        d = Normal(mu, sig)
        lp = d.log_prob(stored_actions[t]).sum(-1).mean(-1)
        new_log_probs.append(lp)
        new_values.append(value)
        new_entropies.append(d.entropy().sum(-1).mean())

    new_log_probs = torch.stack(new_log_probs)  # (T, n_particles)
    new_values = torch.stack(new_values)

    # Importance ratio
    ratio = torch.exp(new_log_probs - old_log_probs)  # (T, n_particles)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (optionally clipped too)
    value_loss = F.mse_loss(new_values, returns)

    # Entropy
    entropy_bonus = torch.stack(new_entropies).mean()

    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
    self.manual_backward(loss)
    self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    opt.step()
    opt.zero_grad()
```

#### 3. Ephemeral State Snapshots for Re-evaluation

PPO requires re-running the **current** policy on the states encountered during the rollout (to compute new log-probs and values). This means we temporarily snapshot these states. **These are discarded after the K epochs — they are NOT a replay buffer.**

For each timestep, store:

```python
stored_states[t] = {
    "pos": particle_population.population.detach().clone(),
    "vel": particle_population.velocity.detach().clone(),
    "pbest": particle_population.pbest.detach().clone(),
    "gbest": particle_population.gbest.detach().clone(),
}
stored_actions[t] = wc1c2.detach().clone()  # the sampled action
```

**Memory concern:** For `n_particles=128, dim=200, T=32`:  
- Each state tensor: `128 × 200 × 4 bytes = 100 KB`  
- 4 state components × 32 steps = `12.8 MB`  
- Actions: `128 × 200 × 3 × 32 = 7.4 MB`  
**Total: ~20 MB** — negligible.

#### 4. tsp_embedding Handling

The GNN embedding is **fixed** per episode (same TSP instance). Compute it once before the rollout and reuse it across all `K_epochs`. Since PPO re-forwards through the policy heads only (not the GNN), and tsp_embedding is passed as a pre-computed tensor, this is automatically handled.

**Important:** For the re-evaluation forward passes in PPO epochs, the GNN is NOT re-run. Only the SwarmEncoder + cross-attention + edge MLPs are re-evaluated with the stored states. The tsp_embedding is a constant context tensor for all epochs.

However, the GNN still needs gradients. Solution: compute `tsp_embedding` with gradients in the **first** PPO epoch's forward passes (it flows through `loss.backward()`). In subsequent epochs, it already has its gradients accumulated.

**Simpler alternative:** Detach tsp_embedding for PPO re-evaluation epochs (it only needs gradients once). Or keep it attached — multiple backwards will accumulate gradients which is fine with zeroed grads between epochs.

---

## Implementation Plan

### Files to Modify

| File | Changes |
|------|---------|
| `nets/tsp_vector.py` | Add `value_head` MLP to `HyperparamTSPModel`; modify `forward()` to return `(mu, sigma, values)` |
| `deep_pso_module.py` | Add `rl_mode` options `"vpg_gae"` and `"ppo"`; refactor training loop into rollout-then-optimise pattern; add `compute_gae()` utility; add PPO loop |
| `configs/tsp.yaml` | Add RL hyperparameters: `rl_mode`, `gamma`, `gae_lambda`, `ppo_epochs`, `ppo_clip`, `value_loss_coeff`, `entropy_coeff` |

### New Config Parameters

```yaml
# RL training
rl_mode: "vpg_gae"        # "reinforce_raw" | "vpg_gae" | "ppo"
gamma: 0.99                # discount factor
gae_lambda: 0.95           # GAE λ
ppo_epochs: 4              # only for PPO: mini-epochs per trajectory
ppo_clip: 0.2              # only for PPO: surrogate clip range
value_loss_coeff: 0.5      # weight of value loss
entropy_coeff: 0.01        # weight of entropy bonus
```

### Step-by-Step

1. **Add `value_head` to `HyperparamTSPModel`**
   - New MLP: `(emb_dim → emb_dim → 1)`  
   - `forward()` now returns `(wc1c2_mu, wc1c2_sigma, values)` where `values: (n_particles,)`
   - Backward-compatible: `reinforce_raw` mode simply ignores the `values` output

2. **Add `compute_gae()` function to `deep_pso_module.py`**
   ```python
   def compute_gae(rewards, values, gamma=0.99, lam=0.95):
       """
       Args:
           rewards: list of T tensors, each (n_particles,)
           values:  list of T tensors, each (n_particles,)
       Returns:
           advantages: list of T tensors
           returns:    list of T tensors
       """
   ```

3. **Refactor `training_step` for VPG+GAE mode**
   - Phase 1: Rollout loop collects `(rewards, values, log_probs, entropies)` per step
   - Phase 2: `compute_gae()`
   - Phase 3: Single loss computation and backward

4. **Add PPO mode to `training_step`**
   - Phase 1: Rollout loop additionally stores `(states, actions, old_log_probs)`
   - Phase 2: `compute_gae()`
   - Phase 3: `K_epochs` of re-evaluation + clipped surrogate + backward

5. **Update `configs/tsp.yaml`** with new hyperparameters

6. **Update `train.py`** — no changes needed (config is passed through `**kwargs`)

### What Does NOT Change
- `pso/particle.py` — untouched, PSO dynamics are the same
- `nets/model.py` — untouched, encoders are the same
- `validation_step` — untouched, validation doesn't use RL loss
- The per-edge `(n_particles, dim, 3)` output shape — untouched

---

## Hyperparameter Recommendations

| Param | REINFORCE (current) | VPG+GAE | PPO |
|-------|--------------------:|--------:|----:|
| `lr` | 3e-4 | 3e-4 | 1e-4 (lower for stability) |
| `gradient_clip` | 1.7 | 1.0 | 0.5 |
| `gamma` | — | 0.99 | 0.99 |
| `gae_lambda` | — | 0.95 | 0.95 |
| `ppo_clip` | — | — | 0.2 |
| `K_epochs` | — | — | 4 |
| `value_loss_coeff` | — | 0.5 | 0.5 |
| `entropy_coeff` | 0.01 | 0.01 | 0.01 |

---

## Expected Impact

| Method | Variance | Bias | Data Efficiency | Compute Cost |
|--------|----------|------|-----------------|--------------|
| REINFORCE + mean baseline | High | None | Low | 1× |
| VPG + GAE | Medium | Low (controllable via λ) | Medium | ~1.1× (value head is tiny) |
| PPO | Medium | Low | High (K reuses) | ~K× forward passes per episode |

**VPG+GAE** is the recommended first upgrade — it adds temporal credit assignment and a learned baseline with minimal compute overhead. If that still isn't enough, switch to **PPO** to squeeze more learning from each episode.
