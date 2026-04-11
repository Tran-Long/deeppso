import os

import hydra
import pytorch_lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml

from envs import EnvDataModule
from logger import CustomLogger
from rl_agents import TSPAgent, TSPACAgent
from rl_algorithms import REINFORCE, Myopic, MyopicI, PPO, VPG

_MODULE_REGISTRY = {
    "TSPAgent": TSPAgent,
    "TSPACAgent": TSPACAgent,

    "Myopic": Myopic,
    "MyopicI": MyopicI,
    "REINFORCE": REINFORCE,
    "PPO": PPO,
    "VPG": VPG,
}


def init_module(config, **kwargs):
    name = config["name"]
    if name not in _MODULE_REGISTRY:
        raise ValueError(
            f"Unknown module '{name}'. Available: {list(_MODULE_REGISTRY)}"
        )
    module_cls = _MODULE_REGISTRY[name]
    module_args = config.get("args", {})
    return module_cls(**module_args, **kwargs)


class GradientNormLogger(L.Callback):
    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Called before the optimizer takes a step (i.e., after the gradients are computed).
        """
        if pl_module.global_step % trainer.log_every_n_steps == 0:
            # 1. Calculate the L2 Norm (without clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                pl_module.parameters(), max_norm=float("inf")
            )

            # 2. Log the value using the Lightning logger
            # Use self.log() to record the metric across all loggers (TensorBoard, etc.)
            pl_module.log(
                "gradient_norm/global_norm", grad_norm, on_step=True, on_epoch=False
            )


class PeriodicTestCallback(L.Callback):
    def __init__(self, run_every_n_epochs=5):
        self.run_every_n_epochs = run_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # Check if it's the right epoch
        if (trainer.current_epoch + 1) % self.run_every_n_epochs == 0:
            # Avoid calling trainer.test() here — it resets trainer._results to None,
            # which breaks the training loop's logger connector assertion that follows.
            # Instead, manually run test_step + on_test_epoch_end without touching trainer state.
            was_training = pl_module.training
            pl_module.eval()
            with torch.no_grad():
                test_dataloaders = trainer.datamodule.test_dataloader()
                pbar_cb = trainer.progress_bar_callback
                # trainer.num_test_batches is a read-only property backed by
                # trainer.test_loop._max_batches; set it so progress bar hooks
                # can read per-dataloader totals, then restore afterward.
                original_max_batches = trainer.test_loop._max_batches
                trainer.test_loop._max_batches = [len(dl) for dl in test_dataloaders]
                # Drive the progress bar callback's lifecycle exactly as Lightning's
                # internal test loop does — this gives correct bar positioning and
                # overwrite behaviour alongside the training bar.
                pbar_cb.on_test_start(trainer, pl_module)
                # TQDMProgressBar: override leave=True so the bar clears on close
                if isinstance(pbar_cb, TQDMProgressBar):
                    pbar_cb.test_progress_bar.leave = False
                for dataloader_idx, dataloader in enumerate(test_dataloaders):
                    for batch_idx, batch in enumerate(dataloader):
                        batch = pl_module.transfer_batch_to_device(
                            batch, pl_module.device, dataloader_idx
                        )
                        pbar_cb.on_test_batch_start(
                            trainer, pl_module, batch, batch_idx, dataloader_idx
                        )
                        pl_module.test_step(
                            batch, batch_idx, dataloader_idx=dataloader_idx
                        )
                        pbar_cb.on_test_batch_end(
                            trainer, pl_module, None, batch, batch_idx, dataloader_idx
                        )
                # RichProgressBar: save the last task id before on_test_end nulls it,
                # then hide it — Rich leaves completed tasks visible by default.
                last_rich_task_id = (
                    pbar_cb.test_progress_bar_id
                    if isinstance(pbar_cb, RichProgressBar)
                    else None
                )
                pbar_cb.on_test_end(trainer, pl_module)
                if last_rich_task_id is not None:
                    pbar_cb.progress.update(last_rich_task_id, visible=False)
                trainer.test_loop._max_batches = original_max_batches
            pl_module.on_test_epoch_end()
            if was_training:
                pl_module.train()


@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Convert OmegaConf to plain dict so downstream code stays unchanged
    config = OmegaConf.to_container(cfg, resolve=True)
    # Start training
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        **config["log"],
    )
    os.makedirs(tensorboard_logger.log_dir, exist_ok=True)
    with open(f"{tensorboard_logger.log_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    csv_logger = pl_loggers.CSVLogger(**config["log"])
    custom_logger = CustomLogger(log_folder=tensorboard_logger.log_dir)
    callbacks = [
        GradientNormLogger(),
        ModelCheckpoint(
            dirpath=f"{tensorboard_logger.log_dir}/checkpoints",
            monitor="val_avg_cost",
            mode="min",
            filename="best-{epoch:02d}-{val_avg_cost:.3f}",
            save_top_k=3,
        )
    ]
    trainer = L.Trainer(
        **config["trainer"], callbacks=callbacks, logger=[tensorboard_logger, csv_logger]
    )
    # decide device for data module from trainer
    device = trainer.accelerator.name() if trainer.accelerator else "cpu"
    # if device == "cuda":
    #     assert (
    #         len(trainer.device_ids) == 1
    #     ), "Multiple devices not supported for data module initialization yet."
    #     device = f"{device}:{trainer.device_ids[0]}" if trainer.device_ids else device
    env_module = EnvDataModule(**config["env"], device=device)
    rl_agent = init_module(config["rl_agent"])
    rl_train = init_module(
        config["rl_train"], agent=rl_agent, custom_logger=custom_logger
    )

    rl_train.val_dataloader_idx2name = (
        env_module.val_dataloader_idx2name
    )  # For logging purposes
    
    hparams_dict = {
        "env": env_module.get_hparams_dict(),
        "rl_agent": config["rl_agent"],
        "rl_train": config["rl_train"],
        "trainer": config["trainer"],
    }
    custom_logger.log_hparams(hparams_dict)
    trainer.fit(rl_train, env_module)
    

if __name__ == "__main__":
    main()
