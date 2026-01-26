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
        n_particles: int | tuple[int, int] | list[int],
        mode: str = "range",
        pso_iterations_train: int = 10,
        net_update_interval: str = "step",  # "step" or "full"
        pso_iterations_infer: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.n_cities = n_cities
        if isinstance(n_cities, int):
            assert isinstance(
                n_particles, int
            ), "n_particles should be an integer when n_cities is an integer."
            self.n_particles = n_particles
        else:
            if mode == "range":
                assert (
                    isinstance(n_cities, tuple) and len(n_cities) == 2
                ), "n_cities should be a tuple of (min_n_cities, max_n_cities) in 'range' mode."
                if isinstance(n_particles, int) or (
                    isinstance(n_particles, tuple | list) and len(n_particles) == 2
                ):
                    self.n_particles = n_particles
                else:
                    raise ValueError(
                        "n_particles should be an integer or a tuple of (min_n_particles, max_n_particles) in 'range' mode."
                    )
            elif mode == "choice":
                assert isinstance(
                    n_cities, list
                ), "n_cities should be a list of choices in 'choice' mode."
                if isinstance(n_particles, int):
                    self.n_particles = n_particles
                elif isinstance(n_particles, list):
                    assert len(n_particles) == len(
                        n_cities
                    ), "n_particles list length should match n_cities list length in 'choice' mode."
                    self.n_particles = {n: p for n, p in zip(n_cities, n_particles)}
            else:
                raise ValueError("mode should be either 'range' or 'choice'.")
        self.mode = mode
        self.pso_iterations_train = pso_iterations_train
        self.net_update_interval = net_update_interval
        self.pso_iterations_infer = pso_iterations_infer

        self.automatic_optimization = False
        self.net = TSPVectorCNNModel()

    def get_n_particles(self, problem: TSPProblem) -> int:
        n_cities = problem.n_cities
        if isinstance(self.n_particles, int):
            return self.n_particles
        elif isinstance(self.n_particles, tuple | list) and len(self.n_particles) == 2:
            return torch.randint(
                self.n_particles[0], self.n_particles[1] + 1, (1,)
            ).item()
        elif isinstance(self.n_particles, dict):
            return self.n_particles.get(n_cities, max(self.n_particles.values()))
        else:
            raise ValueError(
                "n_particles should be an integer, tuple, list, or dict in 'choice' mode."
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-4)
        return optimizer

    def training_step(self, problem: TSPProblem, idx):
        problem_cls = type(problem)
        particle_cls = self.MAPPING_PROBLEM_TO_PARTICLE[problem_cls]
        particle_population = particle_cls(
            n_particles=self.get_n_particles(problem), problem=problem
        )
        # Initial evaluation
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)

        opt = self.optimizers()
        for iter in range(self.pso_iterations_train):
            # Clone and detach population so step() modifications don't affect gradient graph
            input_population = particle_population.population
            wc1c2 = self.net(
                input_population.to(self.device)
            ).cpu()
            particle_population.step(wc1c2, inplace=True)
            solutions, log_probs = particle_population.decode_solutions(
                return_log_probs=True
            )
            costs = problem.evaluate(solutions)

            baseline = costs.mean()
            loss = ((costs - baseline) * log_probs.sum(dim=0)).mean() / solutions.size(0)
            if self.net_update_interval == "step":
                # Update self.net here based on costs and log_probs
                print(f"Iteration {iter+1}, Loss: {loss.item()}")
                self.manual_backward(loss)
                self.clip_gradients(
                    opt, gradient_clip_val=1.5, gradient_clip_algorithm="norm"
                )
                opt.step()
                opt.zero_grad()
            elif self.net_update_interval == "full":
                # Add the gradient accumulation logic here
                loss = loss / self.pso_iterations_train
                self.manual_backward(loss)
            else:
                raise ValueError(
                    "net_update_interval should be either 'step' or 'full'."
                )
            particle_population.population = particle_population.population.detach()
            particle_population.update_metadata(costs)
            
        if self.net_update_interval == "full":
            # Final update for full interval
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
            n_particles=self.get_n_particles(problem), problem=problem
        )
        # Initial evaluation
        solutions = particle_population.decode_solutions(return_log_probs=False)
        costs = problem.evaluate(solutions)
        particle_population.update_metadata(costs)

        for iter in range(self.pso_iterations_infer):
            wc1c2 = self.net(
                particle_population.population.to(self.device).clone().detach()
            ).cpu()
            particle_population.step(wc1c2)
            solutions, log_probs = particle_population.decode_solutions(
                return_log_probs=True
            )
            costs = problem.evaluate(solutions)
            particle_population.update_metadata(costs)

        self.log("val_cost", particle_population.val_gbest, prog_bar=True, batch_size=1)
