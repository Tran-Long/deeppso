import pytorch_lightning as L
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from envs import BaseEnvPSOBatchProblem
from logger import CustomLogger
from rl_agents import TSPAgent
from utils import timeit


class PolicyGradientNaive(L.LightningModule):
    def __init__(
        self,
        agent: TSPAgent,
        pso_iterations_train: int = 10,
        pso_iterations_infer: int = 20,
        pso_using_random: bool = True,
        custom_logger: CustomLogger = None,
    ):
        super().__init__()
        self.agent = agent
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer
        self.pso_using_random = pso_using_random

        self.automatic_optimization = False  # To do manual backward and optimizer step

        # For logging
        self.custom_logger: CustomLogger = custom_logger
        assert (
            self.custom_logger is not None
        ), "CustomLogger instance must be provided for logging population stats during validation."
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_dataloader_idx2name = (
            {}
        )  # To be set by EnvDataModule for logging purposes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=3e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=100, eta_min=1e-5
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, env: BaseEnvPSOBatchProblem, idx):
        observations, _ = env.reset()
        self.log(
            "train_initial_gbest",
            env.val_gbest.mean(),
            prog_bar=True,
        )

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
                wc1c2.detach(),
                temperature=temperature,
                using_random=self.pso_using_random,
            )
            # REINFORCE with per-problem baseline
            # Shapes: reward (B, P), log_probs (B, P), entropy (B, P)
            #   - reward[b,p] = negative stochastic tour cost of particle p in problem b
            #   - log_probs[b,p] = log π(action | state) for particle p, summed over 3 params, meaned over D edges
            #   - baseline[b] = mean reward across particles in problem b
            #   - advantage[b,p] = how much better/worse particle p did vs. swarm average
            baseline = reward.mean(dim=-1, keepdim=True)  # (B, 1)
            advantage = reward - baseline  # (B, P)
            reinforce_loss = -(advantage * log_probs).mean()  # scalar: mean over B×P

            loss = reinforce_loss - 0.01 * entropy.mean()
            # Add to the live graph — backward is deferred until after the loop.
            total_loss = total_loss + loss

        # Normalize by number of iterations so gradient scale is independent of T.
        total_loss = total_loss / self.pso_iterations_train
        # Single backward + optimizer step after all PSO iterations.
        # tsp_embedding's graph is traversed exactly once here.
        self.manual_backward(total_loss)
        self.clip_gradients(opt, gradient_clip_val=1.7, gradient_clip_algorithm="norm")
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.7)
        # self.log("gradient_norm", grad_norm, prog_bar=True, batch_size=1)
        opt.step()
        opt.zero_grad()

        avg_loss = total_loss.detach()
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log(
            "train_gbest",
            env.val_gbest.mean(),
            prog_bar=True,
        )

    def validation_step(self, env: BaseEnvPSOBatchProblem, idx, dataloader_idx=0):
        observations, _ = env.reset()

        self.val_gbest_dataloader["initial"][dataloader_idx] = (
            self.val_gbest_dataloader["initial"].get(dataloader_idx, [])
            + env.val_gbest.tolist()
        )

        # 2. GNN embeddings (single forward pass for all graphs via PyG Batch)
        problem_embeddings = self.agent.get_problem_embedding(env.problem)  # (B, dim, emb_dim)

        # 3. PSO loop
        for pso_idx in range(self.pso_iterations_infer):
            wc1c2, _, _ = self.agent.get_action((*observations, problem_embeddings))

            # Step the env and log population stats for this iteration
            observations, _, _, _, info = env.step_eval(
                wc1c2, using_random=self.pso_using_random
            )
            for i in range(env.batch_size):
                self.custom_logger.log_population_stats(
                    self.val_dataloader_idx2name.get(dataloader_idx, dataloader_idx),
                    idx * env.batch_size + i,
                    pso_idx,
                    info["population_costs"][i],
                    info["gbest_cost"][i],
                )

        self.val_gbest_dataloader["wc1c2"][dataloader_idx] = (
            self.val_gbest_dataloader["wc1c2"].get(dataloader_idx, [])
            + env.val_gbest.tolist()
        )

    def on_validation_epoch_end(self):
        for key in self.val_gbest_dataloader.keys():
            for dataloader_idx in self.val_gbest_dataloader[key].keys():
                val_gbest_list = self.val_gbest_dataloader[key][dataloader_idx]
                avg_val_gbest = sum(val_gbest_list) / len(val_gbest_list)
                self.log(
                    f"val_{key}/{self.val_dataloader_idx2name.get(dataloader_idx, dataloader_idx)}",
                    avg_val_gbest,
                    prog_bar=True,
                )
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}

        self.custom_logger.save_population_stats()
