import torch
import torch.distributions as dist

from utils import timeit

from ..problems import CVRPBatchProblem
from .base import BaseEnvPSOBatchProblem, RewardMode


class CVRPEnv(BaseEnvPSOBatchProblem):
    def __init__(
        self,
        n_particles,
        problem: CVRPBatchProblem,
        reward_mode="greedy",
        patience=5,
        auto_reset=True,
        use_local_search=False,
        **kwargs
    ):
        super().__init__(n_particles, problem, use_local_search, auto_reset, patience)
        self.reward_mode = reward_mode
        self.dim = self.problem.n_cities ** 2

    def initialize_population(self):
        pbest = torch.zeros(
            (self.batch_size, self.n_particles, self.dim),
            dtype=torch.float,
            device=self.device,
        )
        gbest = torch.zeros(
            (self.batch_size, self.dim), dtype=torch.float, device=self.device
        )
        velocity = torch.zeros(
            (self.batch_size, self.n_particles, self.dim),
            dtype=torch.float,
            device=self.device,
        )
        population = torch.rand((self.batch_size, self.n_particles, self.dim))
        return population, velocity, pbest, gbest

    def decode_solutions(self, stochastic: bool=True, temperature: float=1.0):
        '''Decode the population into solutions (routes) and evaluate them
        '''
        device = self.population.device
        B, P, D = self.population.shape
        N = self.problem.n_cities
        BP = B*P
        CA = self.problem.capacity

        bp_indices = torch.arange(BP, device=device)

        def update_visit_masks(visit_masks, actions):
            # visit_masks shape: (BP, N)
            # actions shape: (BP, )
            visit_masks[bp_indices, actions] = 0
            visit_masks[:, 0] = 1
            # depot can be revisited with one exception, 
            # where arrive at depot, and all cities have been visited, 
            # then the route is finished, and the depot should not be visited again
            visit_masks[(actions==0) * (visit_masks[:, 1:]!=0).any(dim=1), 0] = 0
            return visit_masks

        def update_capacity_mask(cur_nodes, used_capacity, demands):
            '''
            Args:
                cur_nodes: shape (BP, )
                used_capacity: shape (BP, )
                demands: flatten + repeat -> shape (BP, N)
            Returns:
                capacity: updated capacity
                capacity_mask: updated mask
            '''
            capacity_mask = torch.ones(size=(BP, N), device=self.device)
            # update capacity
            used_capacity[cur_nodes==0] = 0
            used_capacity = used_capacity + demands[torch.arange(BP), cur_nodes]
            # update capacity_mask
            remaining_capacity = CA - used_capacity # (BP, )
            remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, N) # (BP, N)
            capacity_mask[demand_flat > remaining_capacity_repeat] = 0
            return used_capacity, capacity_mask

        def check_done(visit_mask, actions):
            return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()    
    

        # flat population to (BP, D)
        pop_flat = self.population.view(BP, N, N)
        # demands shape: (B, N) -> (BP, N)
        demand_flat = self.problem.demands.repeat_interleave(P, dim=0) # (BP, N)
        
        starts = torch.zeros((BP, ), dtype=torch.long, device=self.device)
        visit_masks = torch.ones((BP, N), dtype=torch.bool, device=self.device)
        visit_masks = update_visit_masks(visit_masks, starts)
        used_capacity = torch.zeros((BP, ), device=self.device)
        used_capacity, capacity_mask = update_capacity_mask(starts, used_capacity, demand_flat)

        paths = [starts]
        done = check_done(visit_masks, starts)
        prev = starts
        while not done:
            cur_pop = pop_flat[bp_indices, prev, :] # (BP, N)
            combined_mask = visit_masks & capacity_mask.bool() # (BP, N)
            masked_pop = cur_pop.masked_fill(~combined_mask, float('-inf'))
            if stochastic:
                action_dist = dist.Categorical(logits=masked_pop / temperature)
                actions = action_dist.sample() # (BP, )
            else:
                actions = masked_pop.argmax(dim=1) # (BP, )
            visit_masks = update_visit_masks(visit_masks, actions)
            used_capacity, capacity_mask = update_capacity_mask(actions, used_capacity, demand_flat)
            paths.append(actions)
            done = check_done(visit_masks, actions)
            prev = actions
        
        paths = torch.stack(paths, dim=1) # (BP, T)
        paths = paths.view(B, P, -1) # (B, P, T)
        costs = self.problem.evaluate(paths)
        return paths, costs

    def step(
        self,
        wc1c2: torch.Tensor,
        using_random: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        wc1c2 = wc1c2.to(self.device)
        w = wc1c2[..., 0]  # shape: (batch_size, n_particles, dim)
        c1 = wc1c2[..., 1]  # shape: (batch_size, n_particles, dim)
        c2 = wc1c2[..., 2]  # shape: (batch_size, n_particles, dim)
        if using_random:
            r1 = torch.rand((self.batch_size, self.n_particles, 1), device=self.device)
            r2 = torch.rand((self.batch_size, self.n_particles, 1), device=self.device)
            c1 = c1 * r1
            c2 = c2 * r2

        self.velocity = (
            w * self.velocity
            + c1 * (self.pbest - self.population)
            + c2 * (self.gbest.unsqueeze(1) - self.population)
        )
        # Clamp velocity to prevent divergence
        self.velocity = torch.clamp(self.velocity, -4.0, 4.0)
        self.population = self.population + self.velocity

        # Get deterministic cost for metadata update
        _, costs, _, cost_ls, avg_costs = self.decode_solutions_eval()
        delta_val_pbest, delta_val_gbest = self.update_metadata(costs, cost_ls)

        # Update patience counter for early stopping
        improved = (delta_val_gbest > 0)  # (batch_size, )
        self.cnt_patience = torch.where(improved, torch.zeros_like(self.cnt_patience, device=self.device), self.cnt_patience + 1)
        done = self.cnt_patience >= self.patience

        if self.reward_mode == RewardMode.STOCHASTIC:
            _, stochastic_costs = self.decode_solutions(stochastic=True, temperature=temperature)
            reward = -stochastic_costs
        elif self.reward_mode == RewardMode.GREEDY:
            used_costs = cost_ls if self.use_local_search else avg_costs
            reward = -used_costs  # (B, P)
        elif self.reward_mode == RewardMode.PBEST:
            reward = -self.val_pbest  # (B, P)
        elif self.reward_mode == RewardMode.GBEST:
            reward = -self.val_gbest  # (B,)
        elif self.reward_mode == RewardMode.DELTA_PG:
            reward = delta_val_pbest + 0.5 * delta_val_gbest.unsqueeze(-1)  # (B, P)
        elif self.reward_mode == RewardMode.DELTA_GBEST:
            reward = torch.clamp(delta_val_gbest, min=0.0)  # (B,)
        elif self.reward_mode == RewardMode.DELTA_GBEST_RAW:
            reward = delta_val_gbest  # (B,)
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")

        if self.auto_reset:
            # Automatically reset done instances to keep training stable
            # No need to do this for evaluation, as we want to see the final performance after patience runs out
            self._auto_reset_done(done)

        return (
            (self.population, self.velocity, self.pbest, self.gbest, self.problem),
            reward,
            done,
            None,
            {
                "population_costs": costs.cpu()
                .numpy()
                .tolist(),  # List of lists: (batch_size, n_particles)
                "population_costs_ls": cost_ls.cpu()
                .numpy()
                .tolist(),  # List of lists: (batch_size, n_particles)
                "gbest_cost": self.val_gbest.cpu()
                .numpy()
                .tolist(),  # List: (batch_size,)
            },
        )

    def step_train(self, wc1c2, temperature=1.0, using_random=True, **kwargs):
        return self.step(wc1c2, using_random, temperature, **kwargs)
    
    def step_eval(self, wc1c2, using_random=False, **kwargs):
        return self.step(wc1c2, using_random=using_random, **kwargs)