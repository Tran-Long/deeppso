from functools import cached_property
import random
from typing import Optional

import numpy as np
from .base import BaseProblem
import torch
from pathlib import Path
from torch_geometric.data import Data, Batch

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
            elif isinstance(k_sparse, list):
                if len(k_sparse) == 2:
                    offset = (k_sparse[1] - k_sparse[0]) / (n_cities[1] - n_cities[0])
                    return {n: int(k_sparse[0] + offset * (n - n_cities[0])) for n in n_cities_list}
                elif len(k_sparse) == len(n_cities_list):
                    return dict(zip(n_cities_list, k_sparse))
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


# class TSPProblem(BaseProblem):
#     MIN_K_SPARSE = 10
#     def __init__(self, n_cities: int | tuple[int, int] | list[int], k_sparse: Optional[int | list[int]], n_dims=2, mode="range", device="cpu", **kwargs):
#         super().__init__()
#         n_cities_k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)
#         # Randomly choose n_cities value from the mapping for this instance
#         self.n_cities = random.choice(list(n_cities_k_sparse_mapping.keys()))
#         self.k_sparse = n_cities_k_sparse_mapping[self.n_cities]
#         self.n_dims = n_dims
#         self.device = device
#         self.coordinates = torch.rand(self.n_cities, n_dims, device=self.device)
#         self.distance_matrix = torch.norm(self.coordinates[:, None] - self.coordinates, dim=2, p=2)
#         self.distance_matrix[torch.arange(self.n_cities), torch.arange(self.n_cities)] = 1e9  # Prevent zero distance to self

#     @cached_property
#     def pyg_data(self):
#         topk_values, topk_indices = torch.topk(self.distance_matrix, 
#                                            k=self.k_sparse, 
#                                            dim=1, largest=False)
#         edge_index = torch.stack([
#             torch.repeat_interleave(torch.arange(self.n_cities).to(topk_indices.device),
#                                     repeats=self.k_sparse),
#             torch.flatten(topk_indices)
#             ])
#         edge_attr = topk_values.reshape(-1, 1)
#         pyg_data = Data(x=self.coordinates, edge_index=edge_index, edge_attr=edge_attr)
#         return pyg_data

#     @torch.no_grad()
#     def evaluate(self, solutions: torch.Tensor):
#         '''
#         Args:
#             solutions: torch tensor with shape (n_particles, n_cities), each row is a permutation of node indices
#         Returns:
#             costs: torch tensor with shape (n_particles,), cost of each solution
#         '''
#         u = solutions  # shape: (n_particles, n_cities)
#         v = torch.roll(u, shifts=-1, dims=1)  # shape: (n_particles, n_cities)
#         assert (self.distance_matrix[u, v] > 0).all()
#         return torch.sum(self.distance_matrix[u, v], dim=1)

#     @classmethod
#     def from_coordinates(cls, coordinates: torch.Tensor, k_sparse: Optional[int] = None, device="cpu", **kwargs):
#         n_cities = coordinates.shape[0]
#         if k_sparse is None:
#             # k_sparse = max(n_cities // 10, cls.MIN_K_SPARSE)
#             k_sparse = n_cities
#         problem_instance = cls(n_cities=n_cities, k_sparse=k_sparse, n_dims=coordinates.shape[1], device=device)
#         problem_instance.coordinates = coordinates
#         problem_instance.distance_matrix = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
#         problem_instance.distance_matrix[torch.arange(n_cities), torch.arange(n_cities)] = 1e9
#         return problem_instance


class TSPBatchProblem(BaseProblem):
    VAL_DATA_FOLDER = Path(__file__).parents[1] / 'data' / 'tsp'
    def __init__(self, n_cities, k_sparse, batch_size, n_dims=2, mode="range", device="cpu", **kwargs):
        self.batch_size = batch_size
        self.n_cities_k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)
        self.n_cities = random.choice(list(self.n_cities_k_sparse_mapping.keys()))
        self.k_sparse = self.n_cities_k_sparse_mapping[self.n_cities]
        self.n_dims = n_dims
        self.device = device

        self.set_coordinates()

    def set_coordinates(self, coordinates: torch.Tensor = None):
        self.coordinates = coordinates if coordinates is not None else torch.rand(self.batch_size, self.n_cities, self.n_dims, device=self.device)
        self.distance_matrix = torch.norm(self.coordinates[:, :, None] - self.coordinates[:, None, :], dim=3, p=2)
        self.distance_matrix[torch.arange(self.batch_size)[:, None], torch.arange(self.n_cities), torch.arange(self.n_cities)] = 1e9

    @cached_property
    def pyg_data(self):
        batch_size, n_cities, _ = self.coordinates.shape
        topk_values, topk_indices = torch.topk(self.distance_matrix, 
                                            k=self.k_sparse, 
                                            dim=2, largest=False)

        # Vectorized edge_index construction
        # Source nodes: repeat [0,1,...,n_cities-1] k_sparse times for each batch
        src = torch.arange(n_cities, device=self.coordinates.device).unsqueeze(1).expand(-1, self.k_sparse).reshape(-1)
        src = src.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_cities * k_sparse)
        # Destination nodes: flatten topk_indices
        dst = topk_indices.reshape(batch_size, -1)  # (batch_size, n_cities * k_sparse)

        # Add batch offsets to create a single batched graph
        batch_offsets = torch.arange(batch_size, device=self.coordinates.device) * n_cities
        src_batched = (src + batch_offsets[:, None]).reshape(-1)
        dst_batched = (dst + batch_offsets[:, None]).reshape(-1)
        edge_index = torch.stack([src_batched, dst_batched])
        # Flatten node features and edge attributes
        x_batched = self.coordinates.reshape(-1, self.coordinates.size(-1))  # (batch_size * n_cities, n_dims)
        edge_attr_batched = topk_values.reshape(-1, 1)  # (batch_size * n_cities * k_sparse, 1)

        # Create batch assignment tensor for PyG
        batch_tensor = torch.arange(batch_size, device=self.coordinates.device).repeat_interleave(n_cities)

        # Create a single Batch object directly
        batch = Batch(x=x_batched, edge_index=edge_index, edge_attr=edge_attr_batched, batch=batch_tensor)
        return batch

    def evaluate(self, solutions: torch.Tensor):
        '''
        Args:
            solutions: torch tensor with shape (batch_size, n_particles, n_cities), each row is a permutation of node indices
        Returns:
            costs: torch tensor with shape (batch_size, n_particles), cost of each solution
        '''
        batch_size, n_particles, n_cities = solutions.shape
        u = solutions  # shape: (batch_size, n_particles, n_cities)
        v = torch.roll(u, shifts=-1, dims=2)  # shape: (batch_size, n_particles, n_cities)
        batch_indices = torch.arange(batch_size, device=solutions.device).unsqueeze(1).unsqueeze(2).expand(-1, n_particles, -1)
        assert (self.distance_matrix[batch_indices, u, v] > 0).all()
        return torch.sum(self.distance_matrix[batch_indices, u, v], dim=2)
        
    @classmethod
    def get_val_instances(cls, n_cities, k_sparse, batch_size=1, mode="choice", random_include=True, random_size=100, n_dims=2, device="cpu", **kwargs) -> dict[str, list]:
        val_datasets_dict = {}
        n_cities2k_sparse_mapping = get_n_cities_k_sparse_mapping(n_cities, k_sparse, mode)
        print(f">>> n_cities to k_sparse mapping for TSP validation datasets: {n_cities2k_sparse_mapping}")
        all_n_cities = list(set(n_cities2k_sparse_mapping.keys()))
        if n_dims == 2:
            val_files = Path(cls.VAL_DATA_FOLDER).glob('valDataset-*.pt')
            for val_file in val_files:
                n_cities_file = int(val_file.stem.split('-')[-1])
                if n_cities_file not in all_n_cities:
                    continue  # Skip files that don't match the n_cities values we're interested in
                k_sparse_file = n_cities2k_sparse_mapping.get(n_cities_file, n_cities_file)
                val_datasets_dict[f"{n_cities_file}_file"] = []
                # Load and optionally limit the number of instances from the file
                val_tensor = torch.load(val_file)[:random_size]
                for i in range(0, val_tensor.size(0), batch_size):
                    batch_coordinates = val_tensor[i:i+batch_size].to(torch.float).to(device)
                    problem_instance = cls(n_cities=n_cities_file, k_sparse=k_sparse_file, batch_size=batch_size, mode=mode, n_dims=n_dims, device=device)
                    problem_instance.set_coordinates(batch_coordinates)
                    val_datasets_dict[f"{n_cities_file}_file"].append(problem_instance)
            
            test_files = Path(cls.VAL_DATA_FOLDER).glob('tsp*.txt')
            for test_file in test_files:
                n_cities_file = int(test_file.stem.split('_')[0][3:])  # Extract n_cities from filename like 'tsp100_*.txt'
                if n_cities_file not in all_n_cities:
                    continue
                k_sparse_file = n_cities2k_sparse_mapping.get(n_cities_file, n_cities_file)
                val_datasets_dict[f"{n_cities_file}_file_test"] = []
                with open(test_file, 'r') as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), batch_size):
                        batch_coordinates = []
                        for line in lines[i:i+batch_size]:
                            derivate = line.split(' ')
                            coords = torch.tensor(
                                np.array(derivate[0:2*n_cities_file], dtype = np.float64).reshape(n_cities_file, 2)
                            ).float()
                            batch_coordinates.append(coords)
                        batch_coordinates = torch.stack(batch_coordinates).to(device)
                        problem_instance = cls(n_cities=n_cities_file, k_sparse=k_sparse_file, batch_size=batch_size, mode=mode, n_dims=n_dims, device=device)
                        problem_instance.set_coordinates(batch_coordinates)
                        val_datasets_dict[f"{n_cities_file}_file_test"].append(problem_instance)

        need_random = []
        for n_ct in all_n_cities:
            if f"{n_ct}_file" not in val_datasets_dict:
                print(f">>> [Warning]: No validation files found for n_cities in {all_n_cities}. Generating random instances for these.")
                need_random.append(n_ct)
            elif random_include:
                need_random.append(n_ct)

        for n_ct in need_random:
            k_sparse = n_cities2k_sparse_mapping.get(n_ct, n_ct)
            val_datasets_dict[f"{n_ct}_random"] = [cls(n_cities=n_ct, k_sparse=k_sparse, batch_size=batch_size, mode=mode, n_dims=n_dims, device=device) for _ in range(random_size // batch_size + (1 if random_size % batch_size > 0 else 0))]
        print(f">>> Validation datasets for TSP prepared:")
        for name, datasets in val_datasets_dict.items():
            print(f"    - {name}: {len(datasets)} instances")
        return val_datasets_dict