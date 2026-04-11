import pytorch_lightning as L
import torch

from envs import BaseEnv
from logger import CustomLogger
from rl_agents import BaseAgent
from utils import timeit


class BaseRLAlgorithm(L.LightningModule):
    def __init__(
        self,
        agent: BaseAgent,
        pso_iterations_train: int = 10,
        pso_iterations_infer: int = 20,
        max_grad_norm=1.0,
        pso_using_random: bool = True,
        custom_logger: CustomLogger = None,
    ):
        super().__init__()
        self.agent = agent
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer
        self.pso_using_random = pso_using_random
        self.max_grad_norm = max_grad_norm

        self.automatic_optimization = False  # To do manual backward and optimizer step

        # For logging
        self.custom_logger: CustomLogger = custom_logger
        # assert (
        #     self.custom_logger is not None
        # ), "CustomLogger instance must be provided for logging population stats during validation."
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}, "wc1c2_ls": {}}
        self.val_dataloader_idx2name = (
            {}
        )  # To be set by EnvDataModule for logging purposes
        self.test_gbest_dataloader = {"initial": {}, "wc1c2": {}, "wc1c2_ls": {}}
        self.test_dataloader_idx2name = (
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, BaseEnv):
            # move all tensors in your custom data structure to the device
            batch = batch.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def training_step(self, env: BaseEnv, idx):
        raise NotImplementedError(
            "Training step not implemented yet. Focus is on validation loop for now."
        )

    def validation_step(self, env: BaseEnv, idx, dataloader_idx=0):
        # assert (
        #     env.auto_reset == False
        # ), "Validation env should have auto_reset=False to evaluate final performance after patience runs out."
        observations, _ = env.reset()

        self.val_gbest_dataloader["initial"][dataloader_idx] = (
            self.val_gbest_dataloader["initial"].get(dataloader_idx, [])
            + env.val_gbest_ls.tolist()
        )

        # 2. GNN embeddings (single forward pass for all graphs via PyG Batch)
        problem_embeddings = self.agent.get_problem_embedding(
            env.problem
        )  # (B, dim, emb_dim)

        # 3. PSO loop
        for pso_idx in range(self.pso_iterations_infer):
            wc1c2, _, _ = self.agent.get_action((*observations, problem_embeddings))

            # Step the env and log population stats for this iteration
            observations, _, _, _, info = env.step_eval(
                wc1c2, using_random=self.pso_using_random
            )
            # for i in range(env.batch_size):
            #     self.custom_logger.log_population_stats(
            #         self.val_dataloader_idx2name.get(dataloader_idx, dataloader_idx),
            #         idx * env.batch_size + i,
            #         pso_idx,
            #         info["population_costs"][i],
            #         info["gbest_cost"][i],
            #     )

        self.val_gbest_dataloader["wc1c2"][dataloader_idx] = (
            self.val_gbest_dataloader["wc1c2"].get(dataloader_idx, [])
            + env.val_gbest.tolist()
        )
        self.val_gbest_dataloader["wc1c2_ls"][dataloader_idx] = (
            self.val_gbest_dataloader["wc1c2_ls"].get(dataloader_idx, [])
            + env.val_gbest_ls.tolist()
        )
        env.to("cpu")

    def on_validation_epoch_end(self):
        cost_for_checkpoint = {
            "val_avg_cost": [],
            "val_avg_cost_ls": [],
        }
        for key in self.val_gbest_dataloader.keys():
            for dataloader_idx in self.val_gbest_dataloader[key].keys():
                val_gbest_list = self.val_gbest_dataloader[key][dataloader_idx]
                avg_val_gbest = sum(val_gbest_list) / len(val_gbest_list)
                self.log(
                    f"val_{key}/{self.val_dataloader_idx2name.get(dataloader_idx, dataloader_idx)}",
                    avg_val_gbest,
                    prog_bar=True if key == "wc1c2" else False,
                    # sync_dist=True,  # sync across devices for correct avg in DDP
                )
                if key == "wc1c2":
                    cost_for_checkpoint["val_avg_cost"].append(avg_val_gbest)
                if key == "wc1c2_ls":
                    cost_for_checkpoint["val_avg_cost_ls"].append(avg_val_gbest)
        # Log avg cost across all validation dataloaders for checkpointing
        if cost_for_checkpoint["val_avg_cost"]:
            self.log(
                "val_avg_cost",
                sum(cost_for_checkpoint["val_avg_cost"])
                / len(cost_for_checkpoint["val_avg_cost"]),
            )
        if cost_for_checkpoint["val_avg_cost_ls"]:
            self.log(
                "val_avg_cost_ls",
                sum(cost_for_checkpoint["val_avg_cost_ls"])
                / len(cost_for_checkpoint["val_avg_cost_ls"]),
            )
        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}, "wc1c2_ls": {}}

        self.custom_logger.save_population_stats()

    def test_step(self, env: BaseEnv, idx, dataloader_idx=0):
        # # Same as validation step but logs to test_dataloader_idx2name and saves test gbest results separately
        # assert (
        #     env.auto_reset == False
        # ), "Test env should have auto_reset=False to evaluate final performance after patience runs out."
        observations, _ = env.reset()

        self.test_gbest_dataloader["initial"][dataloader_idx] = (
            self.test_gbest_dataloader["initial"].get(dataloader_idx, [])
            + env.val_gbest_ls.tolist()
        )

        # 2. GNN embeddings (single forward pass for all graphs via PyG Batch)
        problem_embeddings = self.agent.get_problem_embedding(
            env.problem
        )  # (B, dim, emb_dim)

        # 3. PSO loop
        for pso_idx in range(self.pso_iterations_infer):
            wc1c2, _, _ = self.agent.get_action((*observations, problem_embeddings))

            # Step the env and log population stats for this iteration
            observations, _, _, _, info = env.step_eval(
                wc1c2, using_random=self.pso_using_random
            )
            # for i in range(env.batch_size):
            #     self.custom_logger.log_population_stats(
            #         self.test_dataloader_idx2name.get(dataloader_idx, dataloader_idx),
            #         idx * env.batch_size + i,
            #         pso_idx,
            #         info["population_costs"][i],
            #         info["gbest_cost"][i],
            #     )

        self.test_gbest_dataloader["wc1c2"][dataloader_idx] = (
            self.test_gbest_dataloader["wc1c2"].get(dataloader_idx, [])
            + env.val_gbest.tolist()
        )
        self.test_gbest_dataloader["wc1c2_ls"][dataloader_idx] = (
            self.test_gbest_dataloader["wc1c2_ls"].get(dataloader_idx, [])
            + env.val_gbest_ls.tolist()
        )
        env.to("cpu")

    def on_test_epoch_end(self):
        for key in self.test_gbest_dataloader.keys():
            for dataloader_idx in self.test_gbest_dataloader[key].keys():
                test_gbest_list = self.test_gbest_dataloader[key][dataloader_idx]
                avg_test_gbest = sum(test_gbest_list) / len(test_gbest_list)
                self.log(
                    f"test_{key}/{self.test_dataloader_idx2name.get(dataloader_idx, dataloader_idx)}",
                    avg_test_gbest,
                    prog_bar=True if key == "wc1c2" else False,
                    # sync_dist=True,  # sync across devices for correct avg in DDP
                )
        self.test_gbest_dataloader = {"initial": {}, "wc1c2": {}, "wc1c2_ls": {}}
