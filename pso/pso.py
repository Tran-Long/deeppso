import torch

from .particle import BaseParticlePopulationModel


class PSOOperator:
    def __init__(self):
        pass

    @classmethod
    @torch.no_grad()
    def evaluate(cls, population: BaseParticlePopulationModel):
        solutions = population.decode_solutions(return_log_probs=False)
        costs = population.problem.evaluate(solutions)

        # Update personal best
        better_pbest_mask = costs < population.val_pbest
        population.val_pbest[better_pbest_mask] = costs[better_pbest_mask]
        population.pbest[better_pbest_mask] = solutions[better_pbest_mask]

        # Update global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost < population.val_gbest:
            population.val_gbest = min_cost
            population.gbest = solutions[min_idx]
        return costs

    @classmethod
    def step(cls, population: BaseParticlePopulationModel, wc1c2: torch.Tensor):
        # wc1c2: shape (n_particles, 3) -> (w, c1, c2), tensor containing model gradients
        w = wc1c2[:, 0].unsqueeze(1)  # shape: (n_particles, 1)
        c1 = wc1c2[:, 1].unsqueeze(1)  # shape: (n_particles, 1)
        c2 = wc1c2[:, 2].unsqueeze(1)  # shape: (n_particles, 1)
        population.population = (
            w * population.population
            + c1 * population.pbest
            + c2 * population.gbest.unsqueeze(0)
        )
