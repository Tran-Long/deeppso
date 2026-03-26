import pytorch_lightning as L
import torch

from .aco_net import Net as ACOGraphNet
from envs.pso import TSPEnvVectorEdgeBatch
from .aco import ACO


class DeepACOModule(L.LightningModule):
    def __init__(
        self,
        n_cities: int | tuple[int, int] | list[int],
        n_ants: int | tuple[int, int] | list[int],
        mode: str = "range",
        aco_iterations_infer: int = 20,
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_cities, int):
            assert isinstance(
                n_ants, int
            ), "n_ants should be an integer when n_cities is an integer."
            self.n_cities2n_ants = {n_cities: n_ants}
        else:
            if mode == "range":
                assert isinstance(n_cities, (tuple, list)) and len(n_cities) > 1, \
                    "n_cities should be a tuple or list with multiple elements in 'range' mode."
                if isinstance(n_ants, int):
                    self.n_cities2n_ants = {n: n_ants for n in range(n_cities[0], n_cities[-1] + 1)}
                elif isinstance(n_ants, (tuple, list)):
                    assert len(n_ants) == len(n_cities), "n_ants range should match n_cities range in 'range' mode."
                    self.n_cities2n_ants = {}
                    for i in range(len(n_cities) - 1):
                        # Create a mapping for each n_cities in the range to a corresponding n_ants value with linear interpolation
                        n_cities_start, n_cities_end = n_cities[i], n_cities[i + 1]
                        n_ants_start, n_ants_end = n_ants[i], n_ants[i + 1]
                        self.n_cities2n_ants[n_cities_start] = n_ants_start
                        self.n_cities2n_ants[n_cities_end] = n_ants_end
                        for n in range(n_cities_start+1, n_cities_end):
                            # Linear interpolation
                            n_ants_n = n_ants_start + (n - n_cities_start) * (n_ants_end - n_ants_start) / (n_cities_end - n_cities_start)
                            self.n_cities2n_ants[n] = int(n_ants_n)
                else:
                    raise ValueError(
                        "n_ants should be an integer or a tuple of (min_n_ants, max_n_ants) in 'range' mode."
                    )
            elif mode == "choice":
                assert isinstance(
                    n_cities, list
                ), "n_cities should be a list of choices in 'choice' mode."
                if isinstance(n_ants, int):
                    self.n_cities2n_ants = {n: n_ants for n in n_cities}
                elif isinstance(n_ants, list):
                    assert len(n_ants) == len(n_cities), "n_particles list length should match n_cities list length in 'choice' mode."
                    self.n_cities2n_ants = {n: p for n, p in zip(n_cities, n_ants)}
            else:
                raise ValueError("mode should be either 'range' or 'choice'.")
        self.mode = mode
        self.aco_iterations_infer = aco_iterations_infer

        self.automatic_optimization = False
        self.net = ACOGraphNet()
        self.eval_metrics = {}
        self.val_idx2names = {}
        self.test_metrics = {}
        self.test_idx2names = {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, env: TSPEnvVectorEdgeBatch, idx):
        assert env.batch_size == 1
        heu_vec = self.net(env.problem.pyg_data.to(self.device))
        heu_mat = self.net.reshape(env.problem.pyg_data.to(self.device), heu_vec) + 1e-9

        aco = ACO(
            n_ants=self.n_cities2n_ants[env.problem.n_cities],
            heuristic=heu_mat,
            distances=env.problem.distance_matrix[0],
            device=self.device,
        )
        opt = self.optimizers()
        costs, log_probs = aco.sample()
        baseline = costs.mean()

        reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
        opt.zero_grad()
        self.manual_backward(reinforce_loss)
        self.log("train_loss", reinforce_loss, batch_size=1)
        opt.step()

    def validation_step(self, env: TSPEnvVectorEdgeBatch, idx, dataloader_idx=0):
        heu_vec = self.net(env.problem.pyg_data.to(self.device)).reshape(env.batch_size, -1)
        heu_mats = reshape_batch(env, heu_vec, self.device) + 1e-9
        for i, heu_mat in enumerate(heu_mats):
            aco = ACO(
                n_ants=self.n_cities2n_ants[env.problem.n_cities],
                heuristic=heu_mat,
                distances=env.problem.distance_matrix[i],
                device=self.device
            )
            costs, _ = aco.sample()
            aco.run(n_iterations=self.aco_iterations_infer)
            baseline = costs.mean()
            best_sample_cost = torch.min(costs)
            best_aco_cost = aco.lowest_cost
            best_aco_path = aco.shortest_path.view(1, 1, -1).cpu()
            best_aco_path_ls = env.problem.local_search(best_aco_path)
            best_aco_cost_ls = env.problem.evaluate(best_aco_path_ls).cpu()[0][0]
            self.eval_metrics[dataloader_idx] = self.eval_metrics.get(dataloader_idx, {"baseline": [], "sample_best": [], "aco_best": [], "aco_best_ls": []})
            self.eval_metrics[dataloader_idx]["baseline"].append(baseline.cpu().item())
            self.eval_metrics[dataloader_idx]["sample_best"].append(best_sample_cost.cpu().item())
            self.eval_metrics[dataloader_idx]["aco_best"].append(best_aco_cost.cpu().item())
            self.eval_metrics[dataloader_idx]["aco_best_ls"].append(best_aco_cost_ls)

    def on_validation_epoch_end(self):
        for dataloader_idx in self.eval_metrics:
            avg_baseline = sum(self.eval_metrics[dataloader_idx]["baseline"]) / len(self.eval_metrics[dataloader_idx]["baseline"])
            avg_sample_best = sum(self.eval_metrics[dataloader_idx]["sample_best"]) / len(self.eval_metrics[dataloader_idx]["sample_best"])
            avg_aco_best = sum(self.eval_metrics[dataloader_idx]["aco_best"]) / len(self.eval_metrics[dataloader_idx]["aco_best"])
            avg_aco_best_ls = sum(self.eval_metrics[dataloader_idx]["aco_best_ls"]) / len(self.eval_metrics[dataloader_idx]["aco_best_ls"])
            self.log(f"val_baseline/{self.val_idx2names.get(dataloader_idx, dataloader_idx)}", avg_baseline)
            self.log(f"val_sample_best/{self.val_idx2names.get(dataloader_idx, dataloader_idx)}", avg_sample_best)
            self.log(f"val_aco_best/{self.val_idx2names.get(dataloader_idx, dataloader_idx)}", avg_aco_best)
            self.log(f"val_aco_best_ls/{self.val_idx2names.get(dataloader_idx, dataloader_idx)}", avg_aco_best_ls)
        self.eval_metrics = {}


    def test_step(self, env: TSPEnvVectorEdgeBatch, idx, dataloader_idx=0):
        heu_vec = self.net(env.problem.pyg_data.to(self.device)).reshape(env.batch_size, -1)
        heu_mats = reshape_batch(env, heu_vec, self.device) + 1e-9
        for i, heu_mat in enumerate(heu_mats):
            aco = ACO(
                n_ants=self.n_cities2n_ants[env.problem.n_cities],
                heuristic=heu_mat,
                distances=env.problem.distance_matrix[i],
                device=self.device
            )
            costs, _ = aco.sample()
            aco.run(n_iterations=self.aco_iterations_infer)
            baseline = costs.mean()
            best_sample_cost = torch.min(costs)
            best_aco_cost = aco.lowest_cost
            best_aco_path = aco.shortest_path.view(1, 1, -1).cpu()
            best_aco_path_ls = env.problem.local_search(best_aco_path)
            best_aco_cost_ls = env.problem.evaluate(best_aco_path_ls).cpu()[0][0]
            self.test_metrics[dataloader_idx] = self.test_metrics.get(dataloader_idx, {"baseline": [], "sample_best": [], "aco_best": [], "aco_best_ls": []})
            self.test_metrics[dataloader_idx]["baseline"].append(baseline.cpu().item())
            self.test_metrics[dataloader_idx]["sample_best"].append(best_sample_cost.cpu().item())
            self.test_metrics[dataloader_idx]["aco_best"].append(best_aco_cost.cpu().item())
            self.test_metrics[dataloader_idx]["aco_best_ls"].append(best_aco_cost_ls)

    def on_test_epoch_end(self):
        for dataloader_idx in self.test_metrics:
            avg_baseline = sum(self.test_metrics[dataloader_idx]["baseline"]) / len(self.test_metrics[dataloader_idx]["baseline"])
            avg_sample_best = sum(self.test_metrics[dataloader_idx]["sample_best"]) / len(self.test_metrics[dataloader_idx]["sample_best"])
            avg_aco_best = sum(self.test_metrics[dataloader_idx]["aco_best"]) / len(self.test_metrics[dataloader_idx]["aco_best"])
            avg_aco_best_ls = sum(self.test_metrics[dataloader_idx]["aco_best_ls"]) / len(self.test_metrics[dataloader_idx]["aco_best_ls"])
            self.log(f"test_baseline/{self.test_idx2names.get(dataloader_idx, dataloader_idx)}", avg_baseline)
            self.log(f"test_sample_best/{self.test_idx2names.get(dataloader_idx, dataloader_idx)}", avg_sample_best)
            self.log(f"test_aco_best/{self.test_idx2names.get(dataloader_idx, dataloader_idx)}", avg_aco_best)
            self.log(f"test_aco_best_ls/{self.test_idx2names.get(dataloader_idx, dataloader_idx)}", avg_aco_best_ls)
        self.test_metrics = {}

def reshape_batch(env, heu_vec, device):
    mat = torch.zeros(
        (env.batch_size, env.n_cities, env.n_cities),
        device=device,
    )

    # Get batched edge_index: (2, batch_size * dim)
    edge_index = env.problem.pyg_data.edge_index

    # Reshape to (2, batch_size, dim) to separate each graph's edges
    edge_index_reshaped = edge_index.view(2, env.batch_size, env.dim)

    # Remove batch offsets to get local node indices (0 to n_cities-1)
    batch_offsets = (
        torch.arange(env.batch_size, device=device) * env.n_cities
    )
    local_src = edge_index_reshaped[0] - batch_offsets[:, None]  # (batch_size, dim)
    local_dst = edge_index_reshaped[1] - batch_offsets[:, None]  # (batch_size, dim)

    # Create batch indices for vectorized assignment
    batch_idx = torch.arange(env.batch_size, device=device).unsqueeze(1).expand(-1, env.dim)  # (batch_size, dim)
    
    # Assign values
    mat[batch_idx, local_src, local_dst] = heu_vec
    return mat