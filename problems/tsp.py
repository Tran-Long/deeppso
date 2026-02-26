from functools import cached_property
import random
from typing import Optional
from .base import BaseProblem
import torch
from pathlib import Path
from torch_geometric.data import Data

def get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode) -> dict[int, int]:
    if isinstance(n_cities, int):
        if k_sparse is None:
            k_sparse = n_cities
        elif not isinstance(k_sparse, int):
            raise ValueError("k_sparse should be an integer or None when n_cities is an integer.")
        return {n_cities: k_sparse}
    else:
        if mode == "range":
            assert len(n_cities) == 2, "n_cities should be a tuple of (min_cities, max_nodes) in 'range' mode."
            n_cities_list = list(range(n_cities[0], n_cities[1]+1))
            if k_sparse is None:
                return {n: n for n in n_cities_list}
            elif isinstance(k_sparse, int):
                return {n: k_sparse for n in n_cities_list}
            else:
                raise ValueError("k_sparse should be an integer or None in 'range' mode.")
        elif mode == "choice":
            assert isinstance(n_cities, list), "n_cities should be a list of choices in 'choice' mode."
            if k_sparse is None:
                return {n: n for n in n_cities}
            elif isinstance(k_sparse, list):
                assert len(k_sparse) == len(n_cities), "k_sparse list length should match n_cities list length in 'choice' mode."
                return dict(zip(n_cities, k_sparse))
            elif isinstance(k_sparse, int):
                return {n: k_sparse for n in n_cities}
            else:
                raise ValueError("k_sparse should be an integer, list, or None in 'choice' mode.")
        else:
            raise ValueError("mode should be either 'range' or 'choice'.")


class TSPProblem(BaseProblem):
    VAL_DATA_FOLDER = Path(__file__).parents[1] / 'data' / 'tsp'
    MIN_K_SPARSE = 10
    def __init__(self, n_cities: int | tuple[int, int] | list[int], k_sparse: Optional[int | list[int]], n_dims=2, mode="range", device="cpu", **kwargs):
        super().__init__()
        n_cities_k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)
        # Randomly choose n_cities value from the mapping for this instance
        self.n_cities = random.choice(list(n_cities_k_sparse_mapping.keys()))
        self.k_sparse = n_cities_k_sparse_mapping[self.n_cities]
        self.n_dims = n_dims
        self.device = device
        self.coordinates = torch.rand(self.n_cities, n_dims, device=self.device)
        self.distance_matrix = torch.norm(self.coordinates[:, None] - self.coordinates, dim=2, p=2)
        self.distance_matrix[torch.arange(self.n_cities), torch.arange(self.n_cities)] = 1e9  # Prevent zero distance to self

    @cached_property
    def pyg_data(self):
        topk_values, topk_indices = torch.topk(self.distance_matrix, 
                                           k=self.k_sparse, 
                                           dim=1, largest=False)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(self.n_cities).to(topk_indices.device),
                                    repeats=self.k_sparse),
            torch.flatten(topk_indices)
            ])
        edge_attr = topk_values.reshape(-1, 1)
        pyg_data = Data(x=self.coordinates, edge_index=edge_index, edge_attr=edge_attr)
        return pyg_data

    @torch.no_grad()
    def evaluate(self, solutions: torch.Tensor):
        '''
        Args:
            solutions: torch tensor with shape (n_particles, n_cities), each row is a permutation of node indices
        Returns:
            costs: torch tensor with shape (n_particles,), cost of each solution
        '''
        u = solutions  # shape: (n_particles, n_cities)
        v = torch.roll(u, shifts=-1, dims=1)  # shape: (n_particles, n_cities)
        assert (self.distance_matrix[u, v] > 0).all()
        return torch.sum(self.distance_matrix[u, v], dim=1)

    @classmethod
    def from_coordinates(cls, coordinates: torch.Tensor, k_sparse: Optional[int] = None, device="cpu", **kwargs):
        n_cities = coordinates.shape[0]
        if k_sparse is None:
            # k_sparse = max(n_cities // 10, cls.MIN_K_SPARSE)
            k_sparse = n_cities
        problem_instance = cls(n_cities=n_cities, k_sparse=k_sparse, n_dims=coordinates.shape[1], device=device)
        problem_instance.coordinates = coordinates
        problem_instance.distance_matrix = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
        problem_instance.distance_matrix[torch.arange(n_cities), torch.arange(n_cities)] = 1e9
        return problem_instance

    
    @classmethod
    def get_val_instances(cls, n_cities, k_sparse, mode, extra_eval_n_cities: Optional[list[int]] = None, extra_eval_k_sparse: Optional[int | list[int]] = None, n_extra_eval: Optional[int] = 100, n_dims: int = 2, device="cpu", **kwargs) -> dict[str, list]:
        val_datasets_dict = {}
        n_cities2k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)

        all_n_cities = list(set(n_cities2k_sparse_mapping.keys()))
        if n_dims == 2:
            val_files = Path(cls.VAL_DATA_FOLDER).glob('valDataset-*.pt')
            for val_file in val_files:
                n_cities = int(val_file.stem.split('-')[-1])
                if n_cities > 100:
                    continue  # Skip large datasets for faster validation
                # Remove from list if present, we will load these from files
                all_n_cities = [n for n in all_n_cities if n != n_cities]
                val_datasets_dict[f"n_{n_cities}_file"] = []
                val_tensor = torch.load(val_file)
                for coordinates in val_tensor:
                    coordinates = coordinates.to(torch.float).to(device)
                    problem_instance = cls.from_coordinates(coordinates, k_sparse=kwargs.get('k_sparse', None), device=device)
                    val_datasets_dict[f"n_{n_cities}_file"].append(problem_instance)
        
        if len(all_n_cities) > 0:
            print(f"Warning: No validation files found for n_cities in {all_n_cities}. Generating random instances for these.")
            for n_cities in all_n_cities:
                k_sparse = n_cities2k_sparse_mapping[n_cities]
                val_datasets_dict[f"n_{n_cities}_random"] = [cls(n_cities=n_cities, k_sparse=k_sparse, n_dims=n_dims, device=device) for _ in range(n_extra_eval)]

        if extra_eval_n_cities is not None:
            assert n_extra_eval is not None, "n_extra must be specified when extra_n_cities is provided."
            if isinstance(extra_eval_k_sparse, int) or extra_eval_k_sparse is None:
                extra_k_sparse_list = [extra_eval_k_sparse] * len(extra_eval_n_cities)
            elif isinstance(extra_eval_k_sparse, list):
                assert len(extra_eval_k_sparse) == len(extra_eval_n_cities), "Length of extra_k_sparse list must match length of extra_n_cities list."
                extra_k_sparse_list = extra_eval_k_sparse
            else:
                raise ValueError("extra_k_sparse should be an integer, list, or None.")
            for n_cities, k_sparse in zip(extra_eval_n_cities, extra_k_sparse_list):
                val_datasets_dict[f"n_{n_cities}_extra"] = [cls(n_cities=n_cities, k_sparse=k_sparse, n_dims=n_dims, device=device) for _ in range(n_extra_eval)]
            
        return val_datasets_dict