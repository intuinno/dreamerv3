from typing import cast
import gymnasium
from .from_gymnasium import FromGymnasium
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import vertebrate_env
import einops
import numpy as np
import os




def stack_camera_obs(obs):
    camera = obs['egocentric_camera']
    camera = einops.rearrange(camera, "(k1  k2) h w c -> (k1 h) (k2 w) c", k1=4)
    obs['egocentric_camera'] = camera
    return obs
    
def last_camera_obs(obs):
    camera = obs['egocentric_camera']
    camera = camera[-1]
    obs['egocentric_camera'] = camera
    return obs
    

class VertebrateEnv(FromGymnasium):
    def __init__(self, stack_camera=True,):
        env = gymnasium.make("vertebrate_env/VertebrateEnv-v0", render_mode="rgb_array", temp_k=16)
        if stack_camera:
            obs_space = env.observation_space
            K, W, H, C = obs_space['egocentric_camera'].shape
            new_space = spaces.Box(0, 255, shape=(W, H, C), dtype=np.uint8)
            obs_space['egocentric_camera'] = new_space
            env = TransformObservation(env, last_camera_obs, obs_space)
        super().__init__(env=env)