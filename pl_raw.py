import pytorch_lightning as L
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from envs import BaseEnvPSOProblem
from rl_agents import TSPAgent


class PolicyGradientNaive(L.LightningModule):
    def __init__(
        self,
        agent: TSPAgent,
        pso_iterations_train: int = 10,
        pso_iterations_infer: int = 20,
        pso_using_random: bool = True,
    ):
        super().__init__()
        self.agent = agent
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer
        self.pso_using_random = pso_using_random

        self.automatic_optimization = False  # To do manual backward and optimizer step

        # For logging
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_dataloader_idx2name = {}  # To be set by EnvDataModule for logging purposes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, env: BaseEnvPSOProblem, idx):
        observations, _ = env.reset()
        self.log(
            "train_initial_gbest",
            env.val_gbest,
            prog_bar=True,
            batch_size=1,
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
            # Raw-cost reward with mean baseline
            baseline = reward.mean()
            reinforce_loss = -((reward - baseline) * log_probs).mean()

            loss = reinforce_loss - 0.01 * entropy
            # Add to the live graph — backward is deferred until after the loop.
            total_loss = total_loss + loss

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
            "train_gbest",
            env.val_gbest,
            prog_bar=True,
            batch_size=1,
        )

    def validation_step(self, envs: list[BaseEnvPSOProblem], idx, dataloader_idx=0):
        B = len(envs)

        # 1. Reset all envs and collect observations
        all_obs = []
        initial_val_gbests = []
        for env in envs:
            obs, _ = env.reset()
            all_obs.append(obs)  # (population, velocity, pbest, gbest, problem)
            initial_val_gbests.append(env.val_gbest)

        self.val_gbest_dataloader["initial"][dataloader_idx] = (
            self.val_gbest_dataloader["initial"].get(dataloader_idx, [])
            + initial_val_gbests
        )

        # 2. Batch GNN embeddings (single forward pass for all graphs via PyG Batch)
        problem_embeddings = self.agent.get_problem_embedding_batch(
            [env.problem for env in envs]
        )  # (B, dim, emb_dim)

        # 3. Batched PSO loop
        k_sparse = envs[0].problem.k_sparse
        for _iter in range(self.pso_iterations_infer):
            # Stack observations for batched forward pass
            batch_pos = torch.stack(
                [obs[0] for obs in all_obs]
            )  # (B, n_particles, dim)
            batch_vel = torch.stack(
                [obs[1] for obs in all_obs]
            )  # (B, n_particles, dim)
            batch_pbest = torch.stack(
                [obs[2] for obs in all_obs]
            )  # (B, n_particles, dim)
            batch_gbest = torch.stack([obs[3] for obs in all_obs])  # (B, dim)

            # Single batched forward pass → (B, n_particles, dim, 3)
            batch_wc1c2 = self.agent.get_action_batch(
                batch_pos,
                batch_vel,
                batch_pbest,
                batch_gbest,
                k_sparse,
                problem_embeddings,
            )

            # Step each env with its slice (env state update is per-env)
            for i, env in enumerate(envs):
                all_obs[i], _, _, _, _ = env.step_eval(
                    batch_wc1c2[i], using_random=self.pso_using_random
                )

        self.val_gbest_dataloader["wc1c2"][dataloader_idx] = self.val_gbest_dataloader[
            "wc1c2"
        ].get(dataloader_idx, []) + [env.val_gbest for env in envs]

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
