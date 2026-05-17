from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class AblationMetric(str, Enum):
    POPULATION_STATS = "population_stats"


class CustomLogger:
    def __init__(self, log_folder):
        self.current_validation_epoch = 0
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.population_stats = {
            "validation_epoch": [],
            "dataloader_key": [],
            "env_idx": [],
            "pso_idx": [],
            "population_costs": [],
            "gbest_costs": [],
        }
        self.avg_gbest_iteration = {}

    def log_hparams(self, hparams_dict):
        with open(self.log_folder / "hparams.yaml", "w") as f:
            yaml.dump(hparams_dict, f)

    def log_population_stats(self, dataloader_key, env_idx, pso_idx, population_costs, gbest_cost):
        assert isinstance(
            population_costs, list
        ), "population_costs should be a list of costs for each particle"
        self.population_stats["validation_epoch"].append(self.current_validation_epoch)
        self.population_stats["dataloader_key"].append(dataloader_key)
        self.population_stats["env_idx"].append(env_idx)
        self.population_stats["pso_idx"].append(pso_idx)
        self.population_stats["population_costs"].append(population_costs)
        self.population_stats["gbest_costs"].append(gbest_cost)

    def save_population_stats(self):
        self.current_validation_epoch += 1
        df = pd.DataFrame(self.population_stats)
        df.to_csv(self.log_folder / "population_stats.csv", index=False)

    def log_avg_gbest_cost(self, dataloader_key, batch_gbest_costs):
        # batch_gbest_costs shape: (n_iter, batch_size)
        if dataloader_key not in self.avg_gbest_iteration:
            self.avg_gbest_iteration[dataloader_key] = np.zeros((len(batch_gbest_costs), 0))
        self.avg_gbest_iteration[dataloader_key] = np.concatenate(
            [self.avg_gbest_iteration[dataloader_key], batch_gbest_costs], axis=1)
    
    def save_avg_gbest_cost(self):
        d = {
            "dataloader_key": [],
        }
        for dataloader_key, costs in self.avg_gbest_iteration.items():
            d["dataloader_key"].append(dataloader_key)
            avg_cost = np.mean(costs, axis=1) # Average over all batches for each iteration
            for i in range(len(avg_cost)):
                d[f"iter{i}"] = d.get(f"iter{i}", []) + [avg_cost[i]]
        df = pd.DataFrame(d)
        df.to_csv(self.log_folder / "avg_gbest_cost_iterations.csv", index=False)