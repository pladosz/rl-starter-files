import argparse
import numpy

import utils
import torch
from utils import device
from utils.agent_debug import Agent_debug

from rew_gen.rew_gen_model import RewGenNet
from rew_gen.RND_model import RNDModelNet
from utils.args_debug import args_debug

class eval:
    def __init__(self,env_name, model, RND_model, rew_gen_model, agent_id, seed = 0,argmax = False, pause  = 0.1, shift = 0, gif_dir = "test_eval", episodes = 3, memory = True, text = False, max_steps_per_episode = 650):
        self.args = args_debug(env_name, model,agent_id, seed,argmax, pause,shift, gif_dir, episodes, memory, text)
        # Set seed for all randomness sources

        utils.seed(self.args.seed)
        # Set device
        # Load environment
        seed = 120
        self.env = utils.make_env(self.args.env_name, seed)
        for _ in range(self.args.shift):
            self.env.seed =120
            self.env.reset()
            self.env.seed =120
        # Load agent
        self.agent =  Agent_debug(self.env.observation_space, self.env.action_space, model,
                    argmax=self.args.argmax, use_memory=self.args.memory, use_text=self.args.text, agent_id = self.args.agent_id)

        #load random rew_gen and rnd, it doesn't matter for visualisation
        rew_gen_model = rew_gen_model
        RND_model = RND_model
        self.episode_count = 0
        # Run the agent
        self.trajectory = torch.zeros((4*max_steps_per_episode+4,1))


    def run(self):
        for episode in range(self.args.episodes):
            self.env.seed =120
            obs = self.env.reset()
            self.env.seed =120
            self.episode_length_counter =0
            while True:
                action = self.agent.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                #add trajectory, actions are necessary otherwise finding key is not rewarded? alternatively we could try key status

                if self.episode_count < 1:
                    agent_position = torch.tensor(self.env.agent_pos)
                    agent_rotation = torch.tensor(self.env.agent_dir).unsqueeze(0)
                    agent_action = torch.tensor(action)
                    agent_state = torch.cat((agent_position,agent_rotation, agent_action)) 
                    step_index = int(self.episode_length_counter)
                    self.trajectory[4*step_index:4*(step_index+1)]=agent_state.unsqueeze(1)
                if done:
                    break
                self.episode_length_counter += 1
            self.episode_count += 1
        return self.trajectory