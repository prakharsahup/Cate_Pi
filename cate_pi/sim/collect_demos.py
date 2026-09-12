import argparse
import os
import torch
import h5py
import numpy as np

# from isaaclab.app import AppLauncher
# For brevity in this skeleton, we assume the env can be launched
from .isaaclab_env import CATEPiEnv, CATEPiEnvCfg
from ..data.force_adverb_labeler import label_episode_force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CATEPi-Grasp-v0")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--output", type=str, default="datasets/catephi_raw.hdf5")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize the Isaac Lab env
    env_cfg = CATEPiEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = CATEPiEnv(cfg=env_cfg)
    
    with h5py.File(args.output, "w") as f:
        data_group = f.create_group("data")
        
        demos_collected = 0
        while demos_collected < args.num_demos:
            obs, _ = env.reset()
            done = False
            
            # Temporary storage for the episode
            ep_tactile_ff = []
            ep_contact_flag = []
            ep_net_force_mag = []
            ep_rgb_wrist = []
            ep_joint_pos = []
            ep_actions = []

            for step in range(env.max_episode_length):
                # Placeholder for scripted policy
                # action = scripted_policy(obs, step)
                action = torch.zeros(args.num_envs, env.action_space.shape[1], device=env.device)
                
                tactile_obs = env._get_tactile_obs()
                
                # Log step data (taking first env for demo purposes)
                ep_tactile_ff.append(tactile_obs["tactile_normal"][0].cpu().numpy()) # Simplified to normal for shape
                ep_contact_flag.append(tactile_obs["contact_flag"][0].cpu().numpy())
                ep_net_force_mag.append(tactile_obs["net_force_mag"][0].cpu().item())
                ep_rgb_wrist.append(obs["policy"]["rgb_wrist"][0].cpu().numpy())
                ep_joint_pos.append(obs["policy"]["joint_pos"][0].cpu().numpy())
                ep_actions.append(action[0].cpu().numpy())

                obs, _, _, _, _ = env.step(action)
                # handle done...
                break # break early for skeleton
            
            # Write to HDF5
            ep_group = data_group.create_group(f"demo_{demos_collected}")
            ep_group.create_dataset("obs/tactile_ff", data=np.stack(ep_tactile_ff))
            ep_group.create_dataset("obs/contact_flag", data=np.stack(ep_contact_flag))
            ep_group.create_dataset("obs/net_force_mag", data=np.stack(ep_net_force_mag))
            ep_group.create_dataset("obs/rgb_wrist", data=np.stack(ep_rgb_wrist))
            ep_group.create_dataset("obs/robot_joint_pos", data=np.stack(ep_joint_pos))
            ep_group.create_dataset("actions", data=np.stack(ep_actions))
            
            # Per-episode language instruction (from force adverb labeller)
            instruction = label_episode_force(ep_net_force_mag)
            ep_group.attrs["language_instruction"] = instruction
            
            demos_collected += 1

if __name__ == "__main__":
    main()
