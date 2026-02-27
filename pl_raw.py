import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pytorch_lightning as L
from envs import BaseEnvPSOProblem
from rl_agents import TSPActorNet

class PolicyGradientNaiveAgent(L.LightningModule):
    def __init__(self, 
            actor: TSPActorNet,
            pso_iterations_train: int = 10,
            pso_iterations_infer: int = 20,
        ):
        super().__init__()
        self.actor = actor
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer

    def get_action(self, obs):
        wc1c2_mu, wc1c2_sigma = self.actor(*obs)
        wc1c2_dist = dist.Normal(wc1c2_mu, wc1c2_sigma)
        wc1c2 = wc1c2_dist.sample()
        log_probs = wc1c2_dist.log_prob(wc1c2)
        pl_entropy = wc1c2_dist.entropy()
        return wc1c2, log_probs, pl_entropy

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )   
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, env: BaseEnvPSOProblem, idx):
        observations, _ = env.reset()
        opt = self.optimizers()
        opt.zero_grad()
        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations
        
        problem_embedding = self.actor.get_problem_embedding(env.problem)

        # Accumulate losses as a live computation graph sum — do NOT call
        # manual_backward inside the loop.  A single backward at the end
        # traverses the tsp_embedding graph exactly once, so there is no
        # "backward through freed graph" error AND the GNN gets its gradients.
        total_loss = torch.zeros(1, device=self.device)
        for _iter in range(self.pso_iterations_train):
            # Temperature annealing: high early (explore) → low late (exploit)
            progress = _iter / max(self.pso_iterations_train - 1, 1)
            temperature = 2.0 * (1.0 - progress) + 0.5 * progress

            wc1c2, log_probs, pl_entropy = self.get_action(
                (*observations, problem_embedding)
            )
            observations, reward, _, _, _ = env.step(wc1c2.detach(), temperature=temperature)
            # Raw-cost reward with mean baseline
            baseline = reward.mean()
            # log_prob per particle: sum over 3 params, mean over dim edges
            # (mean over dim keeps gradient magnitude stable regardless of problem size)
            lp_per_particle = log_probs.sum(dim=-1).mean(dim=-1)  # (n_particles,)
            reinforce_loss = -((reward - baseline) * lp_per_particle).mean()

            # Entropy bonus (mean over dim to keep scale stable)
            entropy = (
                wc1c2_dist.entropy().sum(dim=-1).mean()
            ) 
            loss = reinforce_loss - 0.01 * entropy

            # Add to the live graph — backward is deferred until after the loop.
            total_loss = total_loss + loss

            # --- Greedy multi-start decode for PSO metadata (pbest / gbest) ---
            with torch.no_grad():
                _, costs_greedy = particle_population.decode_solutions_eval()
            particle_population.update_metadata(costs_greedy)

        # Single backward + optimizer step after all PSO iterations.
        # tsp_embedding's graph is traversed exactly once here.
        self.manual_backward(total_loss)
        self.clip_gradients(opt, gradient_clip_val=1.7, gradient_clip_algorithm="norm")
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.7)
        # self.log("gradient_norm", grad_norm, prog_bar=True, batch_size=1)
        opt.step()
        opt.zero_grad()

        avg_loss = total_loss.detach() / self.pso_iterations_train
        self.log("train_loss", avg_loss, prog_bar=True, batch_size=1)
        self.log(
            "train_gbest", particle_population.val_gbest, prog_bar=True, batch_size=1
        )