import pytorch_lightning as L
from torch.utils.data import DataLoader

from .problems import *
from .pso import *

MAPPING_PROBLEM_TO_ENV = {
    "tsp": (TSPBatchProblem, TSPEnvVectorEdgeBatch),
}

class ProblemDataset:
    def __init__(
        self,
        problem_cls: BaseProblem,
        env_cls: BaseEnvPSOBatchProblem,
        n_particles,
        steps_per_epoch: int = 128,
        device="cpu",
        batch_size=1, # default batch = 1 mean spawn a new problem instance at each step, not a batched problem
        **kwargs
    ):
        self.problem_cls = problem_cls
        self.env_cls = env_cls
        self.n_particles = n_particles
        self.device = device
        self.kwargs = kwargs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_problem = self.problem_cls(
            device=self.device, batch_size=self.batch_size, **self.kwargs
        )
        return self.env_cls(
            n_particles=self.n_particles,
            problem=batch_problem,
            device=self.device,
            **self.kwargs,
        )


def single_collate_fn(batch):
    # Since each dataset is a single instance, we just return the instance itself
    return batch[0]


def list_collate_fn(batch):
    """Return a list of envs for batched validation."""
    return batch


class EnvDataModule(L.LightningDataModule):
    def __init__(
        self,
        problem_sig: str,
        n_particles,
        training_cfg:dict,
        validation_cfg:dict,
        test_cfg:dict,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.n_particles = n_particles
        self.training_cfg = training_cfg
        self.validation_cfg = validation_cfg
        self.test_cfg = test_cfg
        self.problem_cls, self.env_cls = MAPPING_PROBLEM_TO_ENV[problem_sig]
        self.train_dataset = ProblemDataset(
            self.problem_cls,
            self.env_cls,
            n_particles=n_particles,
            device=device,
            **training_cfg,
            **kwargs,
        )

        # Prepare validation datasets
        self.val_problems_dict = self.problem_cls.get_val_instances(
            device=device, **validation_cfg, **kwargs
        )
        self.val_datasets_dict = {
            name: [
                self.env_cls(
                    n_particles=self.n_particles,
                    problem=problem,
                    device=self.device,
                    **kwargs,
                )
                for problem in problems
            ]
            for name, problems in self.val_problems_dict.items()
        }
        self.val_dataset_name = list(self.val_datasets_dict.keys())
        self.val_datasets = list(self.val_datasets_dict.values())
        self.val_dataloader_idx2name = {idx: name for idx, name in enumerate(self.val_datasets_dict.keys())}

        # Prepare test datasets if needed. Procedure is the same as validation
        if test_cfg.pop("enable", False):
            self.test_problems_dict = self.problem_cls.get_test_instances(
                device=device, **test_cfg, **kwargs
            )
            self.test_datasets_dict = {
                name: [
                    self.env_cls(
                        n_particles=self.n_particles,
                        problem=problem,
                        device=self.device,
                        **kwargs,
                    )
                    for problem in problems
                ]
                for name, problems in self.test_problems_dict.items()
            }
            self.test_dataset_name = list(self.test_datasets_dict.keys())
            self.test_datasets = list(self.test_datasets_dict.values())
            self.test_dataloader_idx2name = {idx: name for idx, name in enumerate(self.test_datasets_dict.keys())}
        else:
            self.test_problems_dict = {}
            self.test_datasets_dict = {}
            self.test_dataset_name = []
            self.test_datasets = []
            self.test_dataloader_idx2name = {}

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
    
    def test_dataloader(self):
        # For simplicity, we use the same validation datasets for testing. You can modify this to use separate test datasets if needed.
        return [
            DataLoader(dataset, batch_size=1, collate_fn=single_collate_fn)
            for dataset in self.test_datasets
        ]

    def get_hparams_dict(self):
        return {
            "problem_cls": self.problem_cls.__name__,
            "n_particles": self.n_particles,
            "training_cfg": self.training_cfg,
            "validation_cfg": self.validation_cfg,
            "test_cfg": self.test_cfg,
        }