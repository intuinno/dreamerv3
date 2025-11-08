#!/usr/bin/env python3

"""
Simple script to play a trained DreamerV3 agent from a checkpoint.

Usage:
    python play.py --configs defaults play --run.from_checkpoint path/to/checkpoint

The script loads an agent from a checkpoint and plays it continuously in the environment.
The task is automatically loaded from the checkpoint's config.
Rendering is handled by the environment's render function.
Press Ctrl+C to stop.
"""

import pathlib
import sys

# Add the current directory to sys.path
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))

import elements
import embodied
import numpy as np

try:
    import ruamel.yaml as yaml
except ImportError:
    import yaml


def main(argv=None):
    """Main function to load config and play checkpoint."""

    # Load configs from configs.yaml
    dreamerv3_folder = folder / "dreamerv3"
    configs = elements.Path(dreamerv3_folder / "configs.yaml").read()
    configs = yaml.YAML(typ="safe").load(configs)

    # Parse configs argument (no play config needed)
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])

    # Parse remaining arguments
    config = elements.Flags(config).parse(other)

    # Override config for playback mode
    config = config.update(
        {
            "jax": {
                "platform": "cpu",
                "compute_dtype": "float32",
                "prealloc": False,
                "policy_devices": [0],
                "train_devices": [0],
            },
            "run": {"envs": 1, "eval_envs": 1, "train_ratio": 0},
        }
    )

    # Add render mode for play - get the suite from task to configure correctly
    suite = config.task.split("_", 1)[0]
    env_key = f"env.{suite}.render_mode"
    config = config.update({env_key: "human"})

    # Ensure checkpoint is provided
    if not config.run.from_checkpoint:
        raise ValueError("Must provide --run.from_checkpoint argument")

    # Add checkpoint path as model_name for environment display
    checkpoint_path = config.run.from_checkpoint
    model_name_key = f"env.{suite}.model_name"
    config = config.update({model_name_key: checkpoint_path})

    print(f"Playing checkpoint: {config.run.from_checkpoint}")
    print(f"Task: {config.task}")
    print(f"Using configs: {parsed.configs}")

    # Import DreamerV3 components
    from dreamerv3.main import make_agent, make_env
    from functools import partial as bind

    # Create environment and agent
    print("Creating environment...")
    env = make_env(config, 0)

    print("Creating agent...")
    agent = make_agent(config)

    # Load checkpoint
    print("Loading checkpoint...")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(config.run.from_checkpoint, keys=["agent"])
    print("Checkpoint loaded successfully!")

    # Play continuously
    play(env, agent)


def play(env, agent):
    """Main play loop - mimics driver.py logic for single environment.

    Key driver.py logic:
    1. Maintain batched actions with shape (batch_size, ...)
    2. Convert batched actions to single env format
    3. Stack single env obs to batched format
    4. Call policy with batched obs
    5. Apply masking when episode ends
    6. Update reset flag based on is_last
    """

    episode = 0
    total_reward = 0.0
    steps = 0
    batch_size = 1  # Single environment

    # Initialize batched actions like driver does (driver.py line 36-39)
    acts = {
        k: np.zeros((batch_size,) + v.shape, v.dtype) for k, v in env.act_space.items()
    }
    acts["reset"] = np.ones(batch_size, bool)  # Initial reset

    # Initialize policy state (driver.py line 40)
    carry = agent.init_policy(batch_size)

    while True:  # Play continuously
        # Convert batched actions to single env action (driver.py line 59)
        single_act = {k: v[0] for k, v in acts.items()}

        # Step environment (driver.py line 64)
        obs_dict = env.step(single_act)

        # Render the environment
        env.render()

        # Stack single env obs to batched format (driver.py line 65)
        obs = {k: np.stack([v]) for k, v in obs_dict.items()}

        # Separate logs from observations (driver.py line 66-68)
        logs = {k: v for k, v in obs.items() if k.startswith("log/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}

        # Call policy with batched observations (driver.py line 70)
        carry, acts, outs = agent.policy(carry, obs, mode="eval")

        # Apply masking if episode ended (driver.py line 73-75)
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {k: _mask(v, mask) for k, v in acts.items()}

        # Update reset flag (driver.py line 76)
        acts = {**acts, "reset": obs["is_last"].copy()}

        # Track progress
        reward = obs["reward"][0]  # Extract scalar from batch
        is_first = obs["is_first"][0]
        is_last = obs["is_last"][0]

        if is_first:
            if episode > 0:
                print(
                    f"\nEpisode finished: Total Reward = {total_reward:.3f} ({steps} steps)"
                )
                print("=" * 60)
            episode += 1
            print(f"\n{'='*60}")
            print(f"Episode {episode}")
            print("=" * 60)
            total_reward = 0.0
            steps = 0

        total_reward += reward
        steps += 1

        # # Print progress
        # if steps % 10 == 0:
        #     print(
        #         f"  Step {steps:4d} | Reward: {reward:+7.3f} | Total: {total_reward:+8.3f}"
        #     )


def _mask(value, mask):
    """Mask helper function from driver.py line 85-88."""
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    return value * mask.astype(value.dtype)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
