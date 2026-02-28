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
    def get_val_instances(cls, n_cities, k_sparse, mode="choice", random_include=True, random_size=100, n_dims=2, device="cpu", **kwargs) -> dict[str, list]:
        val_datasets_dict = {}
        n_cities2k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)
        all_n_cities = list(set(n_cities2k_sparse_mapping.keys()))
        if n_dims == 2:
            val_files = Path(cls.VAL_DATA_FOLDER).glob('valDataset-*.pt')
            for val_file in val_files:
                n_cities_file = int(val_file.stem.split('-')[-1])
                if n_cities_file not in all_n_cities:
                    continue  # Skip files that don't match the n_cities values we're interested in
                val_datasets_dict[f"{n_cities_file}_file"] = []
                # Load and optionally limit the number of instances from the file
                val_tensor = torch.load(val_file)[:random_size]
                for coordinates in val_tensor:
                    coordinates = coordinates.to(torch.float).to(device)
                    problem_instance = cls.from_coordinates(coordinates, k_sparse=n_cities2k_sparse_mapping.get(n_cities_file, cls.MIN_K_SPARSE), device=device)
                    val_datasets_dict[f"{n_cities_file}_file"].append(problem_instance)
        need_random = []
        for n_ct in all_n_cities:
            if f"{n_ct}_file" not in val_datasets_dict:
                print(f">>> [Warning]: No validation files found for n_cities in {all_n_cities}. Generating random instances for these.")
                need_random.append(n_ct)
            elif random_include:
                need_random.append(n_ct)

        for n_ct in need_random:
            k_sparse = n_cities2k_sparse_mapping.get(n_ct, cls.MIN_K_SPARSE)
            val_datasets_dict[f"{n_ct}_random"] = [cls(n_cities=n_ct, k_sparse=k_sparse, n_dims=n_dims, device=device) for _ in range(random_size)]

        if len(all_n_cities) > 0:
            for n_cities in all_n_cities:
                k_sparse = n_cities2k_sparse_mapping[n_cities]
                val_datasets_dict[f"{n_cities}_random"] = [cls(n_cities=n_cities, k_sparse=k_sparse, n_dims=n_dims, device=device) for _ in range(random_size)]
        print(f">>> Validation datasets for TSP prepared:")
        for name, datasets in val_datasets_dict.items():
            print(f"    - {name}: {len(datasets)} instances")
        return val_datasets_dict