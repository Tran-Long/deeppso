import pytorch_lightning as L

from nets import ACOGraphNet
from problems import ACO, TSPProblem
from pso import *


class DeepACOModule(L.LightningModule):
    def __init__(
        self,
        n_cities: int | tuple[int, int] | list[int],
        n_ants: int | tuple[int, int] | list[int],
        mode: str = "range",
        net_update_interval: str = "step",  # "step" or "full"
        aco_iterations_infer: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.n_cities = n_cities
        if isinstance(n_cities, int):
            assert isinstance(
                n_ants, int
            ), "n_ants should be an integer when n_cities is an integer."
            self.n_ants = n_ants
        else:
            if mode == "range":
                assert (
                    isinstance(n_cities, tuple) and len(n_cities) == 2
                ), "n_cities should be a tuple of (min_n_cities, max_n_cities) in 'range' mode."
                if isinstance(n_ants, int) or (
                    isinstance(n_ants, tuple | list) and len(n_ants) == 2
                ):
                    self.n_ants = n_ants
                else:
                    raise ValueError(
                        "n_ants should be an integer or a tuple of (min_n_ants, max_n_ants) in 'range' mode."
                    )
            elif mode == "choice":
                assert isinstance(
                    n_cities, list
                ), "n_cities should be a list of choices in 'choice' mode."
                if isinstance(n_ants, int):
                    self.n_ants = n_ants
                elif isinstance(n_ants, list):
                    assert len(n_ants) == len(
                        n_cities
                    ), "n_particles list length should match n_cities list length in 'choice' mode."
                    self.n_ants = {n: p for n, p in zip(n_cities, n_ants)}
            else:
                raise ValueError("mode should be either 'range' or 'choice'.")
        self.mode = mode
        self.net_update_interval = net_update_interval
        self.aco_iterations_infer = aco_iterations_infer

        self.automatic_optimization = False
        self.net = ACOGraphNet()
        self.eval_metrics = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, problem: TSPProblem, idx):
        heu_vec = self.net(problem.pyg_data.to(self.device))
        heu_mat = self.net.reshape(problem.pyg_data.to(self.device), heu_vec) + 1e-9

        aco = ACO(
            n_ants=self.n_ants,
            heuristic=heu_mat,
            distances=problem.distance_matrix,
            device=self.device,
        )
        opt = self.optimizers()
        costs, log_probs = aco.sample()
        baseline = costs.mean()
        reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
        opt.zero_grad()
        self.manual_backward(reinforce_loss)
        
        opt.step()
        self.log("train_loss", reinforce_loss, prog_bar=True, batch_size=1)

    def validation_step(self, problem: TSPProblem, idx, dataloader_idx=0):
        heu_vec = self.net(problem.pyg_data.to(self.device))
        heu_mat = self.net.reshape(problem.pyg_data.to(self.device), heu_vec) + 1e-9
        aco = ACO(
            n_ants=self.n_ants,
            heuristic=heu_mat,
            distances=problem.distance_matrix,
            device=self.device
            )
        costs, log_probs = aco.sample()
        aco.run(n_iterations=self.aco_iterations_infer)
        baseline = costs.mean()
        best_sample_cost = torch.min(costs)
        best_aco_cost = aco.lowest_cost
        self.eval_metrics[dataloader_idx] = self.eval_metrics.get(dataloader_idx, {"baseline": [], "sample_best": [], "aco_best": []})
        self.eval_metrics[dataloader_idx]["baseline"].append(baseline.cpu().item())
        self.eval_metrics[dataloader_idx]["sample_best"].append(best_sample_cost.cpu().item())
        self.eval_metrics[dataloader_idx]["aco_best"].append(best_aco_cost.cpu().item())
    
    def on_validation_epoch_end(self):
        for dataloader_idx in self.eval_metrics:
            avg_baseline = sum(self.eval_metrics[dataloader_idx]["baseline"]) / len(self.eval_metrics[dataloader_idx]["baseline"])
            avg_sample_best = sum(self.eval_metrics[dataloader_idx]["sample_best"]) / len(self.eval_metrics[dataloader_idx]["sample_best"])
            avg_aco_best = sum(self.eval_metrics[dataloader_idx]["aco_best"]) / len(self.eval_metrics[dataloader_idx]["aco_best"])
            self.log(f"val_baseline/{dataloader_idx}", avg_baseline, prog_bar=True)
            self.log(f"val_sample_best/{dataloader_idx}", avg_sample_best, prog_bar=True)
            self.log(f"val_aco_best/{dataloader_idx}", avg_aco_best, prog_bar=True)
        self.eval_metrics = {}

