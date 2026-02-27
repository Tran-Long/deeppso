import pytorch_lightning as L
from torch.utils.data import DataLoader

from .problems import *
from .pso import *

MAPPING_PROBLEM_TO_PARTICLE = {
    TSPProblem: TSPEnvVectorEdge,
}


class ProblemDataset:
    def __init__(
        self,
        problem_cls: BaseProblem,
        n_particles,
        step_per_epoch: int = 1000,
        device="cpu",
        **kwargs
    ):
        self.problem_cls = problem_cls
        self.n_particles = n_particles
        self.step_per_epoch = step_per_epoch
        self.device = device
        self.kwargs = kwargs

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        problem = self.problem_cls(device=self.device, **self.kwargs)
        particle_cls = MAPPING_PROBLEM_TO_PARTICLE[self.problem_cls]
        return particle_cls(
            n_particles=self.n_particles,
            problem=problem,
            device=self.device,
            **self.kwargs,
        )


def single_collate_fn(batch):
    # Since each dataset is a single instance, we just return the instance itself
    return batch[0]


class ProblemDataModule(L.LightningDataModule):
    def __init__(
        self,
        problem_cls: BaseProblem,
        n_particles,
        step_per_epoch: int = 128,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.n_particles = n_particles
        self.train_dataset = ProblemDataset(
            problem_cls,
            n_particles=n_particles,
            step_per_epoch=step_per_epoch,
            device=device,
            **kwargs,
        )
        self.val_datasets_dict = problem_cls.get_val_instances(device=device, **kwargs)
        self.val_dataset_name = list(self.val_datasets_dict.keys())
        self.val_datasets = list(self.val_datasets_dict.values())

    def train_dataloader(self):
        # Return a dataloader that yields the same dataset instance repeatedly
        return DataLoader(
            self.train_dataset, batch_size=1, collate_fn=single_collate_fn
        )

    def val_dataloader(self):
        return [
            DataLoader(dataset, batch_size=1, collate_fn=single_collate_fn)
            for dataset in self.val_datasets
        ]
