import pytorch_lightning as L

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
        **kwargs,
    ):
        super().__init__()
        self.n_cities = n_cities
        self.pso_iterations_train = pso_iterations_train
        self.pso_iterations_infer = pso_iterations_infer

        self.automatic_optimization = False
        self.net = HyperparamTSPModel(
            n_particles=n_particles, emb_dim=64, softmax_temperature=10.0
        )
        self.n_particles = n_particles

        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_dataloader_name = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, problem: TSPProblem, idx):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )

        opt = self.optimizers()
        opt.zero_grad()
        pyg_data = problem.pyg_data

        for _iter in range(self.pso_iterations_train):
            wc1c2 = self.net(
                pos=particle_population.population,
                vel=particle_population.velocity,
                pbest=particle_population.pbest,
                gbest=particle_population.gbest,
                tsp_pyg=pyg_data.to(self.device),
            )
            # PSO Update
            particle_population.step(wc1c2)
            solutions, log_probs = particle_population.decode_solutions(
                return_log_probs=True
            )
            costs = problem.evaluate(solutions)

            baseline = costs.mean()

            loss = ((costs - baseline) * log_probs.sum(dim=0)).mean()
            # wc1c2.retain_grad()
            self.manual_backward(loss)

            # Log gradient norm (must be after backward, before step)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), float("inf")
            )
            self.log("gradient_norm", grad_norm, prog_bar=True, batch_size=1)

            # self.clip_gradients(
            #     opt, gradient_clip_val=1.5, gradient_clip_algorithm="norm"
            # )
            opt.step()
            opt.zero_grad()
            # print(f"loss: {loss.item():.6e}")
            # print(f"costs std: {costs.std().item():.6e}")
            # print(
            #     f"log_probs range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]"
            # )
            # print(
            #     f"wc1c2: {torch.abs(wc1c2).mean(dim=0).tolist()}"
            # )  # Check if softmax is saturated
            # print(
            #     f"wc1c2 grad norm: {wc1c2.grad.norm().item() if wc1c2.grad is not None else 'None'}"
            # )
            particle_population.update_metadata(costs)

            self.log("train_loss", loss, prog_bar=True, batch_size=1)
            # exit(0)

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

        for iter in range(self.pso_iterations_infer):
            wc1c2 = self.net(
                pos=particle_population.population,
                vel=particle_population.velocity,
                pbest=particle_population.pbest,
                gbest=particle_population.gbest,
                tsp_pyg=problem.pyg_data.to(self.device),
            )
            particle_population.step(wc1c2)
            # solutions = particle_population.decode_solutions_eval()
            solutions = particle_population.decode_solutions(return_log_probs=False)
            costs = problem.evaluate(solutions)
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
