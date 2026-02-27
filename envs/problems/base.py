from typing import Optional

import pytorch_lightning as L
from torch.utils.data import DataLoader


class BaseProblem:
    # Each instance will be a single data sample, e.g., a TSP instance
    def __init__(self, **kwargs):
        pass

    def evaluate(self, solutions):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @classmethod
    def get_val_instances(cls, device="cpu", **kwargs) -> dict[str, list]:
        raise NotImplementedError("This method should be overridden by subclasses.")




