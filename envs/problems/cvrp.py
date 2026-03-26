import random
import numpy as np
from .base import BaseProblem
import torch
from pathlib import Path
from torch_geometric.data import Data, Batch
import numba as nb
import concurrent.futures
from functools import partial

def get_n_cities(n_cities, mode="range") -> list[int]:
    if isinstance(n_cities, int):
        return [n_cities]
    elif isinstance(n_cities, list):
        if len(n_cities) == 2:
            if mode == "range":
                return list(range(n_cities[0], n_cities[1]+1))
            elif mode == "choice":
                return n_cities
        else:
            return n_cities
    else:
        raise ValueError("n_cities should be either an int or a list of two ints.")

class CVRPBatchProblem(BaseProblem):
    DATA_FOLDER = Path(__file__).parents[1] / 'data' / 'cvrp'
    GOOGLE_SHARED_ID = ""
    def __init__(self, n_cities, batch_size, n_dims=2, mode="range", capacity=50, demand_low=1, demand_high=9, depot_coord=[0.5, 0.5], **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.n_dims = n_dims
        self.capacity = capacity
        self.demand_low = demand_low
        self.demand_high = demand_high
        self.depot_coor = depot_coord

        all_n_cities = get_n_cities(n_cities, mode)
        self.n_cities = random.choice(all_n_cities)
        self.set_coordinates()
        
    def set_coordinates(self, coordinates: torch.Tensor = None, demands: torch.Tensor = None, depot_included: bool = True):
        if coordinates is not None and demands is not None:
            assert coordinates.shape[:2] == demands.shape[:2], "Coordinates and demands should have the same batch size and number of cities."
            self.coordinates = coordinates.to(self.device)
            self.demands = demands.to(self.device)
            if depot_included:
                self.batch_size, self.n_cities, self.n_dims = self.coordinates.shape
                assert torch.all(self.demands[:, 0] == 0), "The first demand should be zero for the depot."
            else:
                self.batch_size, self.n_cities, self.n_dims = self.coordinates.shape
                depot = torch.tensor([self.depot_coor], device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1)
                self.coordinates = torch.cat((depot, self.coordinates), dim=1)
                self.demands = torch.cat((torch.zeros((self.batch_size, 1), device=self.device), self.demands), dim=1)
                self.n_cities += 1 # add depot
        elif coordinates is None and demands is None:
            self.coordinates = torch.rand(size=(self.batch_size, self.n_cities, self.n_dims), device=self.device)
            self.demands = torch.randint(low=self.demand_low, high=self.demand_high+1, size=(self.batch_size, self.n_cities), device=self.device)
            depot = torch.tensor([self.depot_coor], device=self.device).unsqueeze(0).expand(self.batch_size, -1, -1)
            self.coordinates = torch.cat((depot, self.coordinates), dim=1)
            self.demands = torch.cat((torch.zeros((self.batch_size, 1), device=self.device), self.demands), dim=1)
            self.n_cities += 1 # add depot
        else:
            raise ValueError("Both coordinates and demands should be provided together or both should be None.")

        self.distance_matrix = torch.norm(self.coordinates[:, :, None] - self.coordinates[:, None, :], dim=3, p=2)
        self.distance_matrix[torch.arange(self.batch_size, device=self.device)[:, None], torch.arange(self.n_cities, device=self.device), torch.arange(self.n_cities, device=self.device)] = 1e-10

        # Generate pyg_data
        src = torch.arange(self.n_cities, device=self.device).repeat(self.n_cities).repeat(self.batch_size) # (batch_size*n_cities*n_cities,)
        dst = torch.repeat_interleave(torch.arange(self.n_cities, device=self.device), self.n_cities).repeat(self.batch_size) # (batch_size*n_cities*n_cities,)

        batch_offsets = torch.arange(self.batch_size, device=self.device) * self.n_cities
        src = src + batch_offsets.repeat_interleave(self.n_cities * self.n_cities) # (batch_size*n_cities*n_cities,)
        dst = dst + batch_offsets.repeat_interleave(self.n_cities * self.n_cities) # (batch_size*n_cities*n_cities,)

        edge_index = torch.stack((src, dst)) # (2, batch_size*n_cities*n_cities)
        x_batched = self.demands.reshape(-1, 1) # (batch_size*n_cities, 1)
        edge_attr = self.distance_matrix.reshape(-1, 1) # (batch_size*n_cities*n_cities, 1)
        batch_alignment = torch.arange(self.batch_size, device=self.device).repeat_interleave(self.n_cities) # (batch_size*n_cities,)
        self.pyg_data = Batch(x=x_batched, edge_attr=edge_attr, edge_index=edge_index, batch=batch_alignment)

    def to(self, device):
        self.device = device
        # self.coordinates = self.coordinates.to(device)
        # self.demands = self.demands.to(device)
        self.pyg_data = self.pyg_data.to(device)
        return self

    def evaluate(self, solutions):
        '''
        Args:
            solutions: torch tensor with shape (batch_size, n_particles, solution_length), each row is a single solution represented as a sequence of city indices (including depot as 0)
        Returns:
            costs: torch tensor with shape (batch_size, n_particles), cost of each solution
        '''
        batch_size, n_particles, solution_length = solutions.shape
        u = solutions
        v = torch.roll(solutions, shifts=-1, dims=2)
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).unsqueeze(2).expand(-1, n_particles, -1)
        costs = torch.sum(self.distance_matrix[batch_indices, u, v], dim=2) # (batch_size, n_particles)
        return costs


    @classmethod
    def get_val_instances(cls, n_cities, batch_size, mode="range", random_include=False, random_size=100, n_dims=2, capacity=50, demand_low=1, demand_high=9, depot_coord=[0.5, 0.5], **kwargs):
        val_datasets_dict = {}
        all_n_cities = get_n_cities(n_cities, mode)
        if n_dims == 2:
            val_files = list((Path(cls.DATA_FOLDER) / "val").glob("valDataset-*.pt"))
            for val_file in val_files:
                n = int(val_file.stem.split("-")[-1])
                if n not in all_n_cities:
                    continue
                dataset = torch.load(val_file)[:random_size]
                val_datasets_dict[f"{n}_file"] = []
                for i in range(0, dataset.size(0), batch_size):
                    batch_data = dataset[i:i+batch_size]
                    demands = batch_data[:, 0, :]
                    coordinates = batch_data[:, 1:, :]
                    problem_instance = cls(n_cities=n, batch_size=coordinates.size(0), n_dims=n_dims, mode="choice", capacity=capacity, demand_low=demand_low, demand_high=demand_high, depot_coord=depot_coord)
                    problem_instance.set_coordinates(coordinates=coordinates, demands=demands, depot_included=True)
                    val_datasets_dict[f"{n}_file"].append(problem_instance)
        need_random = []
        for n_ct in all_n_cities:
            if f"{n_ct}_file" not in val_datasets_dict:
                print(f"⚠️ [Warning]: No validation files found for {n_ct}. Generating random instances for these.")
                need_random.append(n_ct)
            elif random_include:
                need_random.append(n_ct)

        for n in need_random:
            val_datasets_dict[f"{n}_random"] = [
                cls(n_cities=n, batch_size=batch_size, n_dims=n_dims, mode="choice", capacity=capacity, demand_low=demand_low, demand_high=demand_high, depot_coord=depot_coord)
                for _ in range(random_size // batch_size)
            ]
        for name, datasets in val_datasets_dict.items():
            print(f"    - {name}: {len(datasets)} instances")
        return val_datasets_dict
    
    @classmethod
    def get_test_instances(cls, n_cities, batch_size, mode="range", random_include=False, random_size=100, n_dims=2, capacity=50, demand_low=1, demand_high=9, depot_coord=[0.5, 0.5], **kwargs):
        test_datasets_dict = {}
        all_n_cities = get_n_cities(n_cities, mode)
        if n_dims == 2:
            test_files = list((Path(cls.DATA_FOLDER) / "test").glob("testDataset-*.pt"))
            for test_file in test_files:
                n = int(test_file.stem.split("-")[-1])
                if n not in all_n_cities:
                    continue
                dataset = torch.load(test_file)[:random_size]
                test_datasets_dict[f"{n}_file"] = []
                for i in range(0, dataset.size(0), batch_size):
                    batch_data = dataset[i:i+batch_size]
                    demands = batch_data[:, 0, :]
                    coordinates = batch_data[:, 1:, :]
                    problem_instance = cls(n_cities=n, batch_size=coordinates.size(0), n_dims=n_dims, mode="choice", capacity=capacity, demand_low=demand_low, demand_high=demand_high, depot_coord=depot_coord)
                    problem_instance.set_coordinates(coordinates=coordinates, demands=demands, depot_included=True)
                    test_datasets_dict[f"{n}_file"].append(problem_instance)
        need_random = []
        for n_ct in all_n_cities:
            if f"{n_ct}_file" not in test_datasets_dict:
                print(f"⚠️ [Warning]: No test files found for {n_ct}. Generating random instances for these.")
                need_random.append(n_ct)
            elif random_include:
                need_random.append(n_ct)

        for n in need_random:
            test_datasets_dict[f"{n}_random"] = [
                cls(n_cities=n, batch_size=batch_size, n_dims=n_dims, mode="choice", capacity=capacity, demand_low=demand_low, demand_high=demand_high, depot_coord=depot_coord)
                for _ in range(random_size // batch_size)
            ]
        for name, datasets in test_datasets_dict.items():
            print(f"    - {name}: {len(datasets)} instances")
        return test_datasets_dict

    