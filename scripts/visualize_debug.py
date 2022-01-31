import argparse
import numpy

import utils
from utils import device
from utils.agent_debug import Agent_debug

from rew_gen.rew_gen_model import RewGenNet
from rew_gen.RND_model import RNDModelNet
from utils.args_debug import args_debug

class eval_visualise:
    def __init__(self,env_name, model,agent_id, seed = 0,argmax = False, pause  = 0.1, shift = 0, gif_dir = "test_eval", episodes = 3, memory = True, text = False):
        self.args = args_debug(env_name, model,agent_id, seed,argmax, pause,shift, gif_dir, episodes, memory, text)
        # Set seed for all randomness sources

        utils.seed(self.args.seed)
        # Set device
        print(f"Device: {device}\n")
        # Load environment
        self.env = utils.make_env(self.args.env_name, self.args.seed)
        for _ in range(self.args.shift):
            self.env.reset()
        print("Environment loaded\n")
        # Load agent
        self.agent =  Agent_debug(self.env.observation_space, self.env.action_space, model,
                    argmax=self.args.argmax, use_memory=self.args.memory, use_text=self.args.text, agent_id = self.args.agent_id)
        print("Agent loaded\n")

        #load random rew_gen and rnd, it doesn't matter for visualisation
        rew_gen = RewGenNet(512,device)
        RND_model = RNDModelNet(device)
        # Run the agent


            # Create a window to view the environment
        self.env.render('human')

    def run(self):
        if self.args.gif_dir:
            from array2gif import write_gif
            frames = []
            #add self.agent_id to name
            self.args.gif_dir =self.args.gif_dir+'_{0}'.format(self.args.agent_id)
        for episode in range(self.args.episodes):
            print('episode {0}'.format(episode))
            obs = self.env.reset()

            while True:
                self.env.render('human')
                if self.args.gif_dir:
                    frames.append(numpy.moveaxis(self.env.render("rgb_array"), 2, 0))

                action = self.agent.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                self.agent.analyze_feedback(reward, done)

                if done or self.env.window.closed:
                    break

            if self.env.window.closed:
                break

        if self.args.gif_dir:
            print("Saving gif_dir... ", end="")
            write_gif(numpy.array(frames), self.args.gif_dir+".gif", fps=1/self.args.pause)
            print("Done.")
