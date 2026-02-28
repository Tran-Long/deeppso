from deepaco.deep_aco_module import DeepACOModule
import pytorch_lightning as L
import torch
import yaml
from envs import EnvDataModule, TSPProblem

class GradientNormLogger(L.Callback):
    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        """
        Called before the optimizer takes a step (i.e., after the gradients are computed).
        """
        if pl_module.global_step % trainer.log_every_n_steps == 0:
            # 1. Calculate the L2 Norm (without clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                pl_module.parameters(), 
                max_norm=float('inf') 
            )

            # 2. Log the value using the Lightning logger
            # Use self.log() to record the metric across all loggers (TensorBoard, etc.)
            pl_module.log(
                "gradient_norm/global_norm", 
                grad_norm, 
                on_step=True, 
                on_epoch=False
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train an ACO agent on via Policy Gradient.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    trainer = L.Trainer(
        **config["trainer"],
        callbacks=[GradientNormLogger()])

    # decide device for data module from trainer
    device = trainer.accelerator.name() if trainer.accelerator else "cpu"


    data_module = EnvDataModule(**config["env"], device=device)
    model = DeepACOModule(**config["aco_agent"])

    trainer.fit(model, datamodule=data_module)