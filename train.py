import pytorch_lightning as L
import torch
import yaml

from deep_pso_module import DeepPSOModule
from problems import ProblemDataModule, TSPProblem


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
                pl_module.net.parameters(), max_norm=float("inf")
            )

            # 2. Log the value using the Lightning logger
            # Use self.log() to record the metric across all loggers (TensorBoard, etc.)
            pl_module.log(
                "gradient_norm/global_norm", grad_norm, on_step=True, on_epoch=False
            )


config = yaml.safe_load(open("configs/tsp.yaml", "r"))

model = DeepPSOModule(**config)
config["problem_cls"] = eval(config["problem"])
data_module = ProblemDataModule(**config)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    callbacks=[GradientNormLogger()],
)
trainer.fit(model, datamodule=data_module)
