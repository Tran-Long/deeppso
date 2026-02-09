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
        self.net = HyperparamTSPModel(n_particles=n_particles, emb_dim=64)
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
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)

        opt = self.optimizers()
        opt.zero_grad()
        pyg_data = problem.pyg_data

        for _iter in range(self.pso_iterations_train):
            wc1c2 = self.net(
                pyg_data.to(self.device), particle_population.population.to(self.device)
            )
            # PSO Update
            particle_population.step(wc1c2)
            solutions, log_probs = particle_population.decode_solutions(
                return_log_probs=True
            )
            costs = problem.evaluate(solutions)

            baseline = costs.mean()
            loss = ((costs - baseline) * log_probs.sum(dim=0)).mean()
            self.manual_backward(loss)
            # self.clip_gradients(
            #     opt, gradient_clip_val=1.5, gradient_clip_algorithm="norm"
            # )
            opt.step()
            opt.zero_grad()

            particle_population.population = particle_population.population.detach()
            particle_population.velocity = particle_population.velocity.detach()
            particle_population.update_metadata(costs)

        self.log("train_loss", loss, prog_bar=True, batch_size=1)

    def validation_step(self, problem: TSPProblem, idx, dataloader_idx=0):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )

        wc1c2 = self.net(
            problem.pyg_data.to(self.device),
            particle_population.population.to(self.device),
        )

        # Initial evaluation
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)
        initial_val_gbest = particle_population.val_gbest
        self.val_gbest_dataloader["initial"][dataloader_idx] = (
            self.val_gbest_dataloader["initial"].get(dataloader_idx, [])
            + [initial_val_gbest]
        )

        for iter in range(self.pso_iterations_infer):
            particle_population.step(wc1c2)
            # solutions = particle_population.decode_solutions_eval()
            solutions = particle_population.decode_solutions(return_log_probs=False)
            costs = problem.evaluate(solutions)
            particle_population.update_metadata(costs)
            wc1c2 = self.net(
                problem.pyg_data.to(self.device), particle_population.population
            )

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
