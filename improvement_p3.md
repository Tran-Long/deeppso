# Phase 3 Analysis — Closing the 25–30 % Gap

## Where We Are

| Metric | Value |
|--------|-------|
| TSP-20 optimal | ~3.8–4.2 |
| Current best (128 particles, 32 train / 128 infer) | ~4.8–5.4 |
| Gap to optimal | 25–30 % |
| NN warm-start initial gbest | ~4.2–4.5 |

The system **is learning** — it reliably improves from the initial population. But the final quality plateaus well above optimal. This gap has three separate sources, and the RL upgrade alone will not close all of them.

---

## Root Cause Decomposition

### Source 1: Weak RL Signal (~10 % of the gap)

The raw policy gradient with mean baseline has high variance and no temporal credit assignment. The network doesn't know whether a good reward at step 20 was caused by the action at step 20 or the action at step 5 that set up a good pbest.

**Fix:** VPG + GAE (Option A from `rl_improvements.md`). This is the right first upgrade. GAE with λ=0.95 will give each timestep a properly discounted advantage estimate, and the learned value function will be a far better baseline than `reward.mean()`.

**Expected impact:** Faster, more stable convergence. By itself, this might close 2–5 percentage points of the gap, but the remaining gap comes from fundamental limitations below.

### Source 2: The Decoder Is a Lossy Bottleneck (~10–15 % of the gap)

This is the **largest single contributor** to the remaining gap, and it's independent of the RL method.

The edge-weight vector is a continuous representation of "which edges are promising." The autoregressive greedy decoder converts it to a tour one city at a time. At each step it picks the highest-weight outgoing edge among unvisited cities. This is a **myopic greedy** procedure — it never reconsiders earlier choices.

Consider: for TSP-20 with k_sparse=5, there are 100 edges. A near-optimal tour uses exactly 20 of them. The particle might have learned the correct top-20 edges, but the greedy decoder can lock itself into a suboptimal prefix that makes it impossible to use the remaining good edges. With multi-start (10 starts), you get 10 attempts, but that's still 10 greedy tries on a 20-step sequential decision.

**Evidence:** If the NN warm-start already gives ~4.2–4.5 (which is the nearest-neighbor heuristic + multi-start), and the PSO-optimised particles only reach ~4.8–5.4, the PSO particles' edge weights are *worse* at decoding than a simple heuristic tour converted to edge weights. This means either:
1. The PSO dynamics are corrupting the edge weight structure (likely — velocity updates add noise across all edges), or
2. The decoder can't extract good tours from the continuous vectors PSO produces.

Both are probably true.

**Fixes (pick one or combine):**

#### Fix 2a: 2-opt Local Search Post-Processing (Effort: Low, Impact: High)

After decoding the gbest tour, run a fast 2-opt improvement:

```python
def two_opt(tour, distance_matrix):
    """Iteratively swap pairs of edges if it shortens the tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Cost of removing edges (i-1,i) and (j,j+1) and adding (i-1,j) and (i,j+1)
                d = distance_matrix
                old = d[tour[i-1], tour[i]] + d[tour[j], tour[(j+1) % n]]
                new = d[tour[i-1], tour[j]] + d[tour[i], tour[(j+1) % n]]
                if new < old - 1e-8:
                    tour[i:j+1] = tour[i:j+1].flip(0)
                    improved = True
    return tour
```

For TSP-20, 2-opt runs in microseconds and typically closes 10–20 % of the gap between a greedy/heuristic tour and optimal. This is standard in neural combinatorial optimisation (used by POMO, Attention Model, etc.).

**Where to apply:**  
- **Validation only** (easy, no training change): after the PSO loop, run 2-opt on the gbest tour before logging.
- **In decode_solutions_multistart** (recommended): after greedy decode of each start, run 2-opt. This directly improves pbest/gbest quality, making the PSO's memory much more informative.

**Important:** Do NOT apply 2-opt during the stochastic decode used for REINFORCE — that should remain fast and differentiable-in-spirit. Apply it only to the greedy evaluation path.

#### Fix 2b: Sampling-Based Decode (Effort: Medium, Impact: Medium)

Instead of one greedy decode per start city, do `S` stochastic samples at low temperature and keep the best:

```python
def decode_solutions_sampling(self, n_samples=8, temperature=0.1):
    best_paths = None
    best_costs = torch.full((self.n_particles,), float("inf"), device=self.device)
    for _ in range(n_samples):
        paths = self.decode_solutions(stochastic=True, temperature=temperature)
        costs = self.problem.evaluate(paths)
        improved = costs < best_costs
        if best_paths is None:
            best_paths = paths; best_costs = costs
        else:
            best_paths[improved] = paths[improved]; best_costs[improved] = costs[improved]
    return best_paths, best_costs
```

This gives the decoder more chances to avoid greedy lock-in without any architectural change. Combine with 2-opt on the winner for maximum effect.

### Source 3: PSO Velocity Updates Destroy Edge Structure (~5–10 % of the gap)

The PSO update is:
```
velocity = w * velocity + c1 * (pbest - pos) + c2 * (gbest - pos)
position = position + velocity
```

Even with per-edge control, this is fundamentally a **weighted interpolation** between the current position, pbest, and gbest, plus momentum. After many iterations, the particle positions drift into regions of the 200-dim space that don't correspond to any meaningful tour structure — the edge weights become a smooth-ish blob rather than having the sharp contrast needed for good greedy decoding (tour edges = high, non-tour edges = low).

**Evidence:** The NN warm-start particles start with edge weights of 3.0 (tour) / 0.0 (non-tour) — a sharp, decodable structure. After PSO iterations, all edge weights converge toward similar values as velocity updates smear them.

**Fixes:**

#### Fix 3a: Position Re-Projection (Effort: Low, Impact: Medium)

After each PSO step, re-project the position to maintain sharpness:

```python
def step(self, wc1c2, using_random=False):
    ...  # existing velocity + position update
    # Optional: soft re-projection — scale each city's outgoing edges
    # so the top-k stand out more clearly
    pos = self.population.view(self.n_particles, self.n_cities, self.k_sparse)
    # Per-city softmax with moderate temperature to sharpen edge preferences
    pos = torch.softmax(pos / 0.5, dim=-1) * self.k_sparse  # re-normalise per city
    self.population = pos.view(self.n_particles, self.dim)
```

This preserves the relative ordering of edges per city but prevents all edges from collapsing to similar values.

#### Fix 3b: Periodic Heuristic Re-Seeding (Effort: Low, Impact: Low–Medium)

Every N iterations, decode the current gbest tour and re-initialise the worst K% of particles from that tour + noise:

```python
if _iter % 10 == 0 and _iter > 0:
    worst_k = costs_greedy.argsort(descending=True)[:self.n_particles // 8]
    gbest_tour = ... # decode gbest
    gbest_weights = self._tour_to_edge_weights(gbest_tour)
    particle_population.population[worst_k] = gbest_weights + torch.randn(...) * 0.5
    particle_population.velocity[worst_k] = 0
```

This prevents population stagnation and re-injects decodable structure.

---

## Observation Normalisation (Required for Value Function)

The `rl_improvements.md` plan adds a value function that takes `particle_ctx` as input. For the value function to train stably, the PSO state observations that feed into the SwarmEncoder should be normalised. Currently:

- `pos` values: start at ~0±1 (from randn init) but drift to arbitrary range after velocity updates
- `vel` values: clamped to [-4, 4] but typically much smaller 
- `pbest` values: frozen snapshots of good positions, can be any scale

The value function's MSE loss will be dominated by the raw scale of these inputs.

**Fix:** Add running normalisation to the per-edge MLP input:

```python
# In HyperparamTSPModel.__init__:
self.obs_norm = nn.BatchNorm1d(4)  # normalise the 4 local PSO scalars

# In forward:
# local_feats: (n_particles, dim, 4)
local_feats_flat = local_feats.view(-1, 4)   # (n_particles*dim, 4)
local_feats_flat = self.obs_norm(local_feats_flat)
local_feats = local_feats_flat.view(n_particles, dim, 4)
```

Or simpler — just standardise per-batch:
```python
local_feats = (local_feats - local_feats.mean(dim=(0,1), keepdim=True)) / (local_feats.std(dim=(0,1), keepdim=True) + 1e-8)
```

---

## Implementation Plan (Priority Order)

| Priority | Change | Files | Why | Effort |
|----------|--------|-------|-----|--------|
| **P0** | 2-opt local search on greedy decode | `pso/particle.py` | Closes the decoder bottleneck; biggest single impact on tour quality | Low |
| **P0** | Observation normalisation (BatchNorm on local PSO feats) | `nets/tsp_vector.py` | Required for stable value function training; also helps policy | Trivial |
| **P1** | VPG + GAE (value head + GAE computation + refactored training loop) | `nets/tsp_vector.py`, `deep_pso_module.py`, `configs/tsp.yaml` | Proper temporal credit assignment; learned baseline | Medium |
| **P1** | Log more diagnostics: per-iteration reward, value loss, advantage std, gbest per step | `deep_pso_module.py` | Essential for debugging the RL training | Low |
| **P2** | Sampling-based decode (low-temp stochastic × N, keep best) as alternative/complement to multi-start greedy | `pso/particle.py` | More robust decoding; pairs well with 2-opt | Low |
| **P2** | Position re-projection (per-city softmax after PSO step) | `pso/particle.py` | Maintains decodable edge structure throughout PSO trajectory | Low |
| **P3** | PPO (if VPG+GAE is insufficient) | `deep_pso_module.py` | Squeeze more learning per episode via multi-epoch reuse | Medium |
| **P3** | Periodic heuristic re-seeding of worst particles | `deep_pso_module.py` | Prevents stagnation in long PSO runs | Low |

### What to Implement First

**In a single batch** (all P0 + P1):

1. Add `two_opt()` to `pso/particle.py` and call it inside `decode_solutions_multistart()` on each greedy decode result. This alone should drop the gap from 25–30 % to roughly 10–15 %.

2. Add `nn.BatchNorm1d(4)` on the local PSO features in `HyperparamTSPModel.forward()`.

3. Add `value_head` MLP to `HyperparamTSPModel`, returning `(mu, sigma, values)`.

4. Refactor `training_step` into rollout-then-optimise pattern with GAE (as described in `rl_improvements.md`).

5. Add config params: `rl_mode: "vpg_gae"`, `gamma: 0.99`, `gae_lambda: 0.95`, `value_loss_coeff: 0.5`, `entropy_coeff: 0.01`.

6. Add per-step logging: `train_reward_mean`, `train_value_loss`, `train_advantage_std`.

### Expected Combined Impact

| Stage | Expected gap | Notes |
|-------|-------------|-------|
| Current (REINFORCE + mean baseline) | 25–30 % | Baseline |
| + 2-opt on greedy decode | 10–15 % | Biggest single win |
| + VPG/GAE | 5–10 % | Better credit assignment |
| + sampling decode + position re-projection | 3–7 % | Refinement |
| + PPO (if needed) | 2–5 % | Data efficiency |

The target of **≤5 % gap** (tour cost ≤ 4.2–4.4 for TSP-20) is realistic with P0+P1 changes.
