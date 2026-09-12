import yaml
import h5py
import numpy as np
import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    h5_path = f"{cfg['data_root']}/{cfg['hdf5_name']}"
    
    try:
        h5 = h5py.File(h5_path, "r")
    except FileNotFoundError:
        print(f"Dataset {h5_path} not found. Ensure demo collection has been run.")
        return

    dataset = LeRobotDataset.create(
        repo_id   = "local/catephi",
        fps       = cfg["fps"],
        root      = cfg["output_dir"],
    )

    if "data" not in h5:
        print(f"No 'data' group found in {h5_path}.")
        return

    for ep_key in h5["data"].keys():
        ep = h5["data"][ep_key]
        T  = ep["obs/robot_joint_pos"].shape[0]

        frame = {
            # Standard fields
            "observation.state":         ep["obs/robot_joint_pos"][:],
            "action":                    ep["actions"][:],
            "observation.image.wrist":   ep["obs/rgb_wrist"][:],

            # CATE-π tactile fields
            "observation.tactile_ff":    ep["obs/tactile_ff"][:],     # (T, 20, 25, 3)
            "observation.contact_flag":  ep["obs/contact_flag"][:],   # (T,)

            # Language instruction (same string repeated for all frames)
            "language_instruction": [ep.attrs.get("language_instruction", "Grasp the object.")] * T,
        }
        dataset.add_episode(frame)

    dataset.consolidate()
    print(f"Saved LeRobot dataset to {cfg['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/hdf5_to_lerobot_config.yaml")
    args = parser.parse_args()
    convert(args.config)
