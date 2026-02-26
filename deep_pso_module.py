import pytorch_lightning as L
import torch.distributions as dist

from nets import *
from problems import *
from pso import *


class DeepPSOModule(L.LightningModule):
    MAPPING_PROBLEM_TO_PARTICLE = {
        TSPProblem: TSPVectorEdgePP,
    }

    def __init__(
        self,
        n_cities: int | tuple[int, int] | list[int],
        n_particles: int,
        pso_iterations_train: int = 10,
        pso_iterations_infer: int = 20,
        rl_mode="reinforce_advantage",
        **kwargs,
    ):
        super().__init__()
        self.n_cities = n_cities
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer

        self.automatic_optimization = False
        self.net = HyperparamTSPModel(
            n_particles=n_particles,
            emb_dim=64,
            softmax_temperature=10.0,
        )
        self.n_particles = n_particles
        self.rl_mode = rl_mode  # "reinforce_raw" or "reinforce_advantage"

        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_dataloader_name = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, problem: TSPProblem, idx):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )

        opt = self.optimizers()
        opt.zero_grad()

        # Pre-compute TSP graph embedding once per problem instance.
        # The graph is fixed across all PSO iterations, so there is no need
        # to run the GNN on every iteration — this is ~n_iter speedup for the GNN.
        # Pre-compute TSP graph embedding once per problem instance.
        # Computed WITH gradients so the GNN receives gradient signal.
        # Its graph is freed in the single backward call below, not per-iteration.
        pyg_data = problem.pyg_data.to(self.device)
        tsp_embedding = self.net.tsp_emb(
            pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr
        )  # (dim, emb_dim) — keep full per-edge embeddings

        # Accumulate losses as a live computation graph sum — do NOT call
        # manual_backward inside the loop.  A single backward at the end
        # traverses the tsp_embedding graph exactly once, so there is no
        # "backward through freed graph" error AND the GNN gets its gradients.
        total_loss = torch.zeros(1, device=self.device)
        for _iter in range(self.pso_iterations_train):
            # Temperature annealing: high early (explore) → low late (exploit)
            progress = _iter / max(self.pso_iterations_train - 1, 1)
            temperature = 2.0 * (1.0 - progress) + 0.5 * progress

            wc1c2_mu, wc1c2_sigma = self.net(
                pos=particle_population.population,
                vel=particle_population.velocity,
                pbest=particle_population.pbest,
                gbest=particle_population.gbest,
                tsp_embedding=tsp_embedding,  # reuse cached embedding
                k_sparse=problem.k_sparse,  # pass k_sparse for graph-aware pooling
            )
            wc1c2_dist = dist.Normal(wc1c2_mu, wc1c2_sigma)
            # PSO Update — per-edge hyperparameters
            wc1c2 = wc1c2_dist.sample()  # (n_particles, dim, 3)
            log_probs = wc1c2_dist.log_prob(wc1c2)  # (n_particles, dim, 3)
            particle_population.step(wc1c2, using_random=True)

            # --- Stochastic decode for REINFORCE gradient ---
            solutions_stochastic = particle_population.decode_solutions(
                stochastic=True, temperature=temperature
            )
            costs_stochastic = problem.evaluate(solutions_stochastic)

            if self.rl_mode == "reinforce_advantage":
                

            elif self.rl_mode == "reinforce_raw":
                # Raw-cost reward with mean baseline
                reward = -costs_stochastic.detach()  # (n_particles,)
                baseline = reward.mean()
                # log_prob per particle: sum over 3 params, mean over dim edges
                # (mean over dim keeps gradient magnitude stable regardless of problem size)
                lp_per_particle = log_probs.sum(dim=-1).mean(dim=-1)  # (n_particles,)
                reinforce_loss = -((reward - baseline) * lp_per_particle).mean()
            else:
                raise ValueError(f"Unsupported rl_mode: {self.rl_mode}")

            # Entropy bonus (mean over dim to keep scale stable)
            entropy = (
                wc1c2_dist.entropy().sum(dim=-1).mean()
            )  # mean over (particles, dim)
            loss = reinforce_loss - 0.01 * entropy

            # Add to the live graph — backward is deferred until after the loop.
            total_loss = total_loss + loss

            # --- Greedy multi-start decode for PSO metadata (pbest / gbest) ---
            with torch.no_grad():
                _, costs_greedy = particle_population.decode_solutions_multistart()
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

    def validation_step(self, problem: TSPProblem, idx, dataloader_idx=0):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )

        initial_val_gbest = particle_population.val_gbest
        self.val_gbest_dataloader["initial"][dataloader_idx] = (
            self.val_gbest_dataloader["initial"].get(dataloader_idx, [])
            + [initial_val_gbest]
        )

        # Cache TSP embedding for validation too — full per-edge
        pyg_data_val = problem.pyg_data.to(self.device)
        with torch.no_grad():
            tsp_embedding_val = self.net.tsp_emb(
                pyg_data_val.x, pyg_data_val.edge_index, pyg_data_val.edge_attr
            )  # (dim, emb_dim)

        for iter in range(self.pso_iterations_infer):
            wc1c2_mu, wc1c2_sigma = self.net(
                pos=particle_population.population,
                vel=particle_population.velocity,
                pbest=particle_population.pbest,
                gbest=particle_population.gbest,
                tsp_embedding=tsp_embedding_val,
                k_sparse=problem.k_sparse,
            )
            wc1c2_dist = dist.Normal(wc1c2_mu, wc1c2_sigma)
            wc1c2 = wc1c2_dist.sample()  # (n_particles, dim, 3)
            particle_population.step(wc1c2, using_random=True)
            # solutions = particle_population.decode_solutions_eval()
            _, costs = particle_population.decode_solutions_multistart()
            particle_population.update_metadata(costs)

        self.val_gbest_dataloader["wc1c2"][dataloader_idx] = self.val_gbest_dataloader[
            "wc1c2"
        ].get(dataloader_idx, []) + [particle_population.val_gbest]

    def on_validation_epoch_end(self):
        for key in self.val_gbest_dataloader.keys():
            for dataloader_idx in self.val_gbest_dataloader[key].keys():
                val_gbest_list = self.val_gbest_dataloader[key][dataloader_idx]
                avg_val_gbest = sum(val_gbest_list) / len(val_gbest_list)
                self.log(
                    f"val_{key}/{self.val_dataloader_name.get(dataloader_idx, dataloader_idx)}",
                    avg_val_gbest,
                    prog_bar=True,
                )
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
