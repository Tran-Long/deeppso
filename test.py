import hydra
import pytorch_lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
import yaml
from pathlib import Path
from envs import EnvDataModule
from logger import CustomLogger
from rl_agents import TSPAgent, TSPACAgent
from rl_algorithms import REINFORCE, Myopic, MyopicI, PPO, VPG
import pandas as pd
import os
import yaml

_MODULE_REGISTRY = {
    "TSPAgent": TSPAgent,
    "TSPACAgent": TSPACAgent,

    "Myopic": Myopic,
    "MyopicI": MyopicI,
    "REINFORCE": REINFORCE,
    "PPO": PPO,
    "VPG": VPG,
}

def init_module(config, **kwargs):
    name = config["name"]
    if name not in _MODULE_REGISTRY:
        raise ValueError(
            f"Unknown module '{name}'. Available: {list(_MODULE_REGISTRY)}"
        )
    module_cls = _MODULE_REGISTRY[name]
    module_args = config.get("args", {})
    return module_cls(**module_args, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the trained model.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing the trained model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to run the test after training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
    )
    args = parser.parse_args()

    config_path = f"{args.folder}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    test_folder = Path(args.folder) / "test_results"
    if not args.force and \
        os.path.isdir(test_folder):
        print(f"Test results already exist in {test_folder}. Skipping test.")
        exit(0)
    else:
        os.makedirs(test_folder, exist_ok=True)

    device = args.device
    if not device:
        trainer = L.Trainer(**config["trainer"])
        device = trainer.accelerator.name() if trainer.accelerator else "cpu"
    print(f">>> Using device: {device}")

    
    env_module = EnvDataModule(**config["env"], device=device)

    checkpoint_dir = os.path.join(args.folder, "checkpoints")
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt") and not f.startswith("epoch=")
    ]
    for ckpt_file in checkpoint_files:
        csv_logger = pl_loggers.CSVLogger(
            save_dir=test_folder.parent, 
            name=test_folder.name,
            version=ckpt_file.split(".ckpt")[0],
        )
        trainer = L.Trainer(
            **config["trainer"], logger=csv_logger
        )
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]

        rl_agent = init_module(config["rl_agent"])
        
        rl_train = init_module(
            config["rl_train"], agent=rl_agent,
        )
        rl_train.load_state_dict(state_dict, strict=True)
        rl_train.test_dataloader_idx2name = (
            env_module.test_dataloader_idx2name
        )
        # best_model = rl_train.__class__.load_from_checkpoint(ckpt_path, agent=rl_agent)
        print(">>> Testing checkpoint:", ckpt_file)
        trainer.test(rl_train, datamodule=env_module)