import hydra
import pytorch_lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from omegaconf import DictConfig, OmegaConf

from envs import EnvDataModule
from logger import CustomLogger
from pl_raw import PolicyGradientNaive
from rl_agents import TSPAgent

_MODULE_REGISTRY = {
    "TSPAgent": TSPAgent,
    "PolicyGradientNaive": PolicyGradientNaive,
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


@hydra.main(config_path="configs", config_name="tsp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Convert OmegaConf to plain dict so downstream code stays unchanged
    config = OmegaConf.to_container(cfg, resolve=True)

    # Start training
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        **config["log"],
    )
    custom_logger = CustomLogger(log_folder=tensorboard_logger.log_dir)
    trainer = L.Trainer(
        **config["trainer"],
        callbacks=[GradientNormLogger()],
        logger=tensorboard_logger,
    )
    # decide device for data module from trainer
    device = trainer.accelerator.name() if trainer.accelerator else "cpu"

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
