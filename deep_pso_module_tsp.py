import pytorch_lightning as L

from nets import *
from problems import *
from pso import *


class DeepPSOModule(L.LightningModule):
    MAPPING_PROBLEM_TO_PARTICLE = {
        TSPProblem: TSPParticleVector,
    }

    def __init__(
        self,
        n_cities: int | tuple[int, int] | list[int],
        n_particles: int,
        mode: str = "range",
        pso_iterations_train: int = 10,
        net_update_interval: str = "step",  # "step" or "full"
        pso_iterations_infer: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.n_cities = n_cities
        self.mode = mode
        self.pso_iterations_train = pso_iterations_train
        self.net_update_interval = net_update_interval
        self.pso_iterations_infer = pso_iterations_infer

        self.automatic_optimization = False
        self.net = TSPVectorGraph(n_particles=n_particles)
        self.n_particles = n_particles

        self.val_gbest_dataloader = {"initial": {}, "wc1c2": {}}
        self.val_dataloader_name = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, problem: TSPProblem, idx):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )
        

        opt = self.optimizers()
        pyg_data = problem.pyg_data
        population, wc1c2 = self.net(pyg_data.to(self.device))
        particle_population.population = population.detach().clone()
        # Initial evaluation
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)

        # PSO Update
        particle_population.step(wc1c2)
        solutions, log_probs = particle_population.decode_solutions(
            return_log_probs=True
        )
        costs = problem.evaluate(solutions)

        baseline = costs.mean()
        loss = ((costs - baseline) * log_probs.sum(dim=0)).mean() / solutions.size(0)
        self.manual_backward(loss)
        self.clip_gradients(
            opt, gradient_clip_val=1.5, gradient_clip_algorithm="norm"
        )
        opt.step()
        opt.zero_grad()
        
            
        self.log("train_loss", loss, prog_bar=True, batch_size=1)

    def validation_step(self, problem: TSPProblem, idx, dataloader_idx=0):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.n_particles, problem=problem, device=self.device
        )

        population, wc1c2 = self.net(problem.pyg_data.to(self.device))
        particle_population.population = population.detach().clone()

        # Initial evaluation
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)
        initial_val_gbest = particle_population.val_gbest
        self.val_gbest_dataloader["initial"][dataloader_idx] = self.val_gbest_dataloader["initial"].get(dataloader_idx, []) + [initial_val_gbest]

        for iter in range(self.pso_iterations_infer):
            particle_population.step(wc1c2)
            solutions = particle_population.decode_solutions(
                return_log_probs=False
            )
            costs = problem.evaluate(solutions)
            particle_population.update_metadata(costs)

        self.val_gbest_dataloader["wc1c2"][dataloader_idx] = self.val_gbest_dataloader["wc1c2"].get(dataloader_idx, []) + [particle_population.val_gbest]

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

