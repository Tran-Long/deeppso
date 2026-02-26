# Deep PSO for TSP — Phase 2 Analysis

## Context

After implementing all P0–P3 fixes from `improvement.md`, the system shows marginal improvement:

- `pso_iterations_train=16` → best cost ≈ **5.9**
- `pso_iterations_train=64` → best cost ≈ **5.5**

For TSP-20 (random points in [0,1]²) the expected optimal tour length is approximately **3.8–4.2**. A purely random tour averages **5.5–6.5**. So the system is producing near-random tours even after 64 PSO iterations guided by a trained neural network.

The two user observations:

1. **Initial evaluation stays the same every epoch** — this is structurally expected because each validation call creates a fresh random population. But it also reveals that *the network is learning almost nothing transferable across episodes*.
2. **More iterations help only marginally** — this is the critical clue. It means the PSO dynamics themselves are fundamentally flawed, not just under-iterated.

---

## Root Cause Analysis

### 1. Stochastic Decoding Corrupts PSO's Memory (pbest / gbest)

This is the **single most damaging issue** remaining.

PSO stores `pbest` (the position that gave each particle its best-ever cost) and `gbest` (the swarm's best-ever position). The core PSO assumption is: **positions near good positions are also good**. This is the foundation of the cognitive and social terms.

Currently, during training:
```python
solutions = particle_population.decode_solutions(stochastic=True)
costs = problem.evaluate(solutions)
particle_population.update_metadata(costs)   # pbest/gbest updated based on stochastic costs
```

A particle at position `x` gets a stochastic tour that happens to be good due to **lucky sampling**. Its position `x` is stored as `pbest`. But `x` isn't inherently a "good" position — a different sample from the same position might produce a terrible tour. When other particles are pulled toward this `pbest` via `c1 * (pbest - x)`, they're chasing noise.

**Evidence:** This explains why more iterations help little — particles are being attracted to positions that aren't actually better, just luckier. The PSO converges to noise, not signal.

**Fix — Dual-decode strategy:**
```python
# Use stochastic decode for REINFORCE gradient computation
solutions_stochastic = particle_population.decode_solutions(stochastic=True)
costs_stochastic = problem.evaluate(solutions_stochastic)
# ... compute REINFORCE loss using costs_stochastic ...

# Use greedy decode for pbest/gbest update — this gives a deterministic,
# repeatable mapping from position to tour quality.
with torch.no_grad():
    solutions_greedy = particle_population.decode_solutions(stochastic=False)
    costs_greedy = problem.evaluate(solutions_greedy)
particle_population.update_metadata(costs_greedy)
```

This way:
- REINFORCE gets the variance it needs (stochastic) to compute gradients.
- PSO metadata (pbest, gbest) reflects the deterministic quality of positions — if a position consistently decodes to a good tour, it's a genuinely good position.

### 2. Random Start City Injects Irrecoverable Noise

```python
start = torch.randint(low=0, high=self.n_cities, size=(self.n_particles,), ...)
```

Every decode call picks a fresh random starting city for each particle. This means:
- The **same position** evaluated twice can give very different costs.
- `pbest` quality depends on which starting city was sampled when `pbest` was set.
- Greedy decode isn't even deterministic across calls (different start → different tour).

**Fix — Fix the start city per particle within an episode:**
```python
# In initialize_population:
self.start_cities = torch.randint(low=0, high=self.n_cities, size=(self.n_particles,), device=self.device)

# In decode_solutions:
start = self.start_cities  # Not re-randomized
```

This makes greedy decode fully deterministic for a given particle position, which is essential for pbest/gbest to be meaningful signals.

**Even better — multi-start decode for evaluation:**
```python
def decode_solutions_multistart(self, n_starts=5):
    best_paths = None
    best_costs = torch.full((self.n_particles,), float('inf'), device=self.device)
    for _ in range(n_starts):
        paths = self.decode_solutions(stochastic=False)  # random start but greedy
        costs = self.problem.evaluate(paths)
        improved = costs < best_costs
        if best_paths is None:
            best_paths = paths
            best_costs = costs
        else:
            best_paths[improved] = paths[improved]
            best_costs[improved] = costs[improved]
    return best_paths, best_costs
```

### 3. The Improvement Reward Decays to Zero

```python
prev_costs = particle_population.val_pbest.clone().detach()
# ...loop...
improvement = (prev_costs - costs).detach()
prev_costs = torch.min(prev_costs, costs.detach())
```

`prev_costs` is the running best per particle. After 2–3 iterations of stochastic decoding, the lucky-best cost is already quite low. At iteration 10+, `improvement = prev_costs - costs ≈ 0` for most particles, and the reward signal is pure noise.

**Evidence:** This directly explains observation 2 — adding more iterations doesn't help because later iterations see zero reward. The network gets meaningful gradient only from the first few steps.

**Fix A — Use the final absolute cost (with separate baseline):**
Instead of per-step improvement, treat the entire PSO trajectory as one episode and use the **final gbest** as the reward:
```python
# After the PSO loop ends:
final_reward = -particle_population.val_gbest  # negated because lower cost = better
# Distribute this reward to all (iteration, particle) log_probs using REINFORCE
```

This gives every iteration credit for contributing to the final outcome.

**Fix B — Reward-to-go (discounted future improvement):**
```python
# After collecting all per-step rewards:
rewards = [...]  # shape: (n_iterations, n_particles)
gamma = 0.99
returns = []
R = 0
for r in reversed(rewards):
    R = r + gamma * R
    returns.insert(0, R)
# Use returns[t] as the reward for iteration t
```

This ensures later iterations still get signal from the improvements they contribute to.

**Fix C (simplest) — Use raw cost with running exponential baseline:**
Drop the per-step improvement idea entirely. Use raw cost each iteration, but with a proper running baseline:
```python
# Per-step:
reward = -costs  # negative cost = reward
baseline = reward.mean()
loss = -((reward - baseline) * log_probs.sum(-1)).mean()
```

Every iteration gets a gradient signal because absolute costs are never zero.

### 4. The Action Space Is Too Constrained (3 Scalars Per Particle)

The network predicts `(w, c1, c2)` — three scalar hyperparameters per particle. The directions `(pbest - x)` and `(gbest - x)` are **fixed by the PSO formula**. The network can only control **how much** to follow each pre-set direction, not **where** to go.

For PSO in a 200-dimensional space, this is like having a car with only a gas pedal and no steering wheel. The pre-set directions (toward pbest and gbest) may not point toward good tours at all.

**Evidence:** This is why the network struggles — even with perfect (w, c1, c2) predictions, the best it can do is balance between three crude forces in 200 dimensions.

**Fix A — Predict per-edge hyperparameters:**
Instead of 3 scalars per particle, predict `(w_i, c1_i, c2_i)` per edge dimension. Shape: `(n_particles, dim, 3)`. This gives the network much finer control — it can emphasize cognitive pull on some edges and social pull on others.

This requires changing the network output layer and the PSO step:
```python
# In step():
# w, c1, c2 now have shape (n_particles, dim)
self.velocity = w * self.velocity + c1 * (self.pbest - self.population) + c2 * (self.gbest - self.population)
```

The network head would produce `(n_particles, dim, 3)` instead of `(n_particles, 3)`. This is more expensive but far more expressive.

**Fix B — Neural velocity perturbation (recommended):**
Keep the standard PSO velocity as a base, but let the network add a learned perturbation:
```python
# Standard PSO
v_pso = w * self.velocity + c1 * (pbest - x) + c2 * (gbest - x)
# Neural network predicts an additional velocity shift per particle
v_neural = net.predict_velocity_delta(x, v, pbest, gbest, tsp_emb)  # (n_particles, dim)
# Combined
self.velocity = v_pso + alpha * v_neural
```

The network can now suggest entirely new search directions — not limited to pbest/gbest. The standard PSO provides a warm-start direction, and the network refines it.

For REINFORCE, sample `v_neural` from a distribution (e.g. Gaussian with predicted mean and sigma, similar to current w/c1/c2), and use the log_prob in the policy gradient.

**Fix C — Replace PSO with direct neural update:**
Go further: eliminate the PSO formula entirely and let the network predict the full next position:
```python
x_new = net(x, v, pbest, gbest, tsp_emb)  # (n_particles, dim)
```

This turns the problem into a sequence-to-sequence model (positions → next positions) trained with REINFORCE. The PSO formula is just one particular architecture for this, and it may not be the best.

### 5. The Network Doesn't Know the Edge Structure

The `ParticleVectorStem` processes the 200-dim particle vector (which represents edge weights) with a generic MLP that mean/max pools across all dimensions:

```python
x = torch.stack(args, dim=-1)   # (n_particles, 200, 3)
x = self.stem(x)                # (n_particles, 200, emb_dim)
x_mean = x.mean(dim=1)          # (n_particles, emb_dim)  ← structure lost
```

Dimension 0 of the particle vector corresponds to edge (node_0 → node_0's 1st nearest neighbor), dimension 1 to edge (node_0 → node_0's 2nd nearest neighbor), etc. These are NOT interchangeable — they have specific graph-structural meaning. Mean-pooling treats them all the same.

**Fix — Reshape and use graph structure:**
```python
# Reshape particle vectors to (n_particles, n_cities, k_sparse, ...)
x = x.view(n_particles, n_cities, k_sparse, -1)

# Option A: Pool per-city, then process
x = x.mean(dim=2)  # (n_particles, n_cities, emb_dim) — one vector per city per particle
# Then use a GNN or attention over cities

# Option B: Scatter onto edge_index and use a GNN
# Map the dim=200 back to edges using the same edge_index as TSP graph,
# then run a lightweight GNN to get particle-aware edge embeddings
```

This gives the network awareness of which edges belong to which cities.

### 6. No Heuristic Initialization (Warm-Start)

All 256 particles start as random Gaussian vectors, producing random tours. The PSO has to discover good tours from scratch every episode. For TSP-20, a simple **nearest-neighbor heuristic** gives tours of quality ~4.5–5.0. If some particles started near heuristic-quality positions, the PSO would have much better pbest/gbest to work with from the start.

**Fix — Nearest-neighbor seeded initialization:**
```python
def initialize_population(self):
    ...
    population = []
    # Seed a fraction with nearest-neighbor heuristic
    n_heuristic = self.n_particles // 4  # 25% from heuristic
    for i in range(n_heuristic):
        nn_tour = self._nearest_neighbor_tour(start_city=i % self.n_cities)
        edge_weights = self._tour_to_edge_weights(nn_tour)
        particle = edge_weights + torch.randn_like(edge_weights) * 0.3
        population.append(particle)
    # Rest are random
    for _ in range(self.n_particles - n_heuristic):
        particle = torch.randn(self.dim, ...)
        population.append(particle)
    return torch.stack(population)

def _nearest_neighbor_tour(self, start_city=0):
    """Simple NN heuristic: always visit the nearest unvisited city."""
    visited = {start_city}
    tour = [start_city]
    current = start_city
    for _ in range(self.n_cities - 1):
        dist = self.problem.distance_matrix[current].clone()
        dist[list(visited)] = float('inf')
        next_city = dist.argmin().item()
        visited.add(next_city)
        tour.append(next_city)
        current = next_city
    return torch.tensor(tour, device=self.device)

def _tour_to_edge_weights(self, tour):
    """Convert a tour (permutation) to high edge weights for used edges."""
    weights = torch.zeros(self.dim, device=self.device)
    edge_index = self.problem.pyg_data.edge_index
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        # Find which index in edge_index has this (u,v) edge
        mask = (edge_index[0] == u) & (edge_index[1] == v)
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0][0]
            weights[idx] = 3.0  # High weight for edges in the tour
    return weights
```

### 7. Temperature Annealing During PSO Iterations

Even for the REINFORCE stochastic decode, the temperature should anneal during the PSO trajectory:
- **Early iterations:** high temperature → more exploration, discover diverse tour structures.
- **Late iterations:** low temperature → exploit the good positions found, refine solutions.

```python
for _iter in range(self.pso_iterations_train):
    # Anneal from temperature 2.0 down to 0.5 over iterations
    progress = _iter / max(self.pso_iterations_train - 1, 1)
    temperature = 2.0 * (1 - progress) + 0.5 * progress

    # In decode_solutions, divide logits by temperature:
    solutions = particle_population.decode_solutions(stochastic=True, temperature=temperature)
```

And in `decode_solutions`:
```python
def decode_solutions(self, stochastic=False, temperature=1.0):
    ...
    if stochastic:
        actions = dist.Categorical(logits=masked_mat / temperature).sample()
    ...
```

### 8. `k_sparse` Configuration Mismatch for Validation

The validation instances are loaded via `from_coordinates`, which sets `k_sparse = n_cities` (fully connected):
```python
@classmethod
def from_coordinates(cls, coordinates, k_sparse=None, device="cpu"):
    if k_sparse is None:
        k_sparse = n_cities  # ← Full graph
```

But training uses `k_sparse=10`. The network trains on 200-dim particles but validates on 400-dim particles (20×20). **This is a train/val mismatch.**

**Fix:** Pass the training `k_sparse` to `from_coordinates`:
```python
problem_instance = cls.from_coordinates(coordinates, k_sparse=10, device=device)
```

Or better, propagate the config's `k_sparse` to the validation loader.

---

## Recommended Action Plan (Priority Order)

| Priority | Change | Why | Effort |
|----------|--------|-----|--------|
| **P0** | Fix k_sparse val mismatch (§8) | Network sees different dim at val vs train | Trivial |
| **P0** | Greedy decode for pbest/gbest, stochastic only for REINFORCE (§1) | PSO memory is currently noise | Low |
| **P0** | Fix start city per particle (§2) | Makes greedy decode deterministic | Trivial |
| **P1** | Switch to raw-cost reward with mean baseline (§3, Fix C) | Improvement reward dies after few steps | Low |
| **P1** | Temperature annealing in decode (§7) | Balance explore/exploit | Low |
| **P2** | Nearest-neighbor warm start (§6) | Much better starting point for PSO | Medium |
| **P2** | Graph-aware particle encoder (§5) | Current encoder destroys structure | Medium |
| **P3** | Neural velocity perturbation (§4, Fix B) | Unconstrain the action space | High |

**Start with P0 changes.** They fix fundamental correctness issues — the current system can't possibly work well because pbest/gbest are noise and val uses a different particle dimension. These alone should yield a significant jump from 5.5 to somewhere in the 4.x range.

Then add P1 (reward signal + temperature), P2 (warm start + encoder), and finally P3 (architecture) as needed.
