from enum import Enum
from pathlib import Path

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
