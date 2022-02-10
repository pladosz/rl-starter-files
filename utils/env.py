import gym
import gym_minigrid
from gym_minigrid.wrappers import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = ReseedWrapper(env, seeds = [120], seed_idx = 0)
    #env.seed(seed)
    return env
