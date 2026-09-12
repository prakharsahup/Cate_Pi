import numpy as np
import yaml
import os

def calibrate_contact_thresholds(env, num_episodes=50):
    """
    Collect net_force_mag distributions from known-contact and
    known-no-contact frames. Fit p_thresh and n_thresh from the
    resulting empirical CDFs.
    """
    contact_forces    = []
    no_contact_forces = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        for step in range(env.max_episode_length):
            # scripted policy: close gripper on object (known contact)
            # or hold arm in air (known no-contact)
            # action = scripted_policy(obs, step)
            # obs, _, done, info = env.step(action)

            # Mock data for skeleton
            phase = "contact" if step > env.max_episode_length / 2 else "free"
            
            # This should come from the env's contact sensor
            # f = obs["net_force_mag"]
            f = np.random.normal(loc=5.0 if phase == "contact" else 0.1, scale=0.5)

            if phase == "contact":
                contact_forces.append(f)
            else:
                no_contact_forces.append(f)

    p_thresh = np.percentile(no_contact_forces, 95)   # 95th pctile of no-contact
    n_thresh = 3   # require at least 3 taxels above threshold (tune from data)

    print(f"p_thresh={p_thresh:.4f}  n_thresh={n_thresh}")
    
    # Write to configs/tactile_sensor.yaml
    config_path = os.path.join(os.path.dirname(__file__), "../configs/tactile_sensor.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f_out:
        yaml.dump({"p_thresh": float(p_thresh), "n_thresh": int(n_thresh)}, f_out)
        
    return p_thresh, n_thresh
