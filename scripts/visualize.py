import argparse
import numpy

import utils
from utils import device

from rew_gen.rew_gen_model import RewGenNet
from rew_gen.RND_model import RNDModelNet
from rew_gen.episodic_buffer import Episodic_buffer
import torch

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=True,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--agent_id", type=int, default=0,
                    help="agent id to run visualisation on")
parser.add_argument("--best", action = "store_true", default=False,
                    help="deteremine if run the best agent")
parser.add_argument("--update", type=int, default=60,
                    help="determine from which update to take best agent")

args = parser.parse_args()
# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.seed(2150)
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text, agent_id = args.agent_id, best = args.best, update = args.update)
print("Agent loaded\n")

#load random rew_gen and rnd, it doesn't matter for visualisation
rew_gen = RewGenNet(363,device)
RND_model = RNDModelNet(device)
# Run the agent
episodic_buffer = Episodic_buffer()

if args.gif:
   from array2gif import write_gif
   frames = []
   #add agent_id to name
   if args.best:
        args.gif =args.gif+'_best_u_{0}'.format(args.update)     
   else:
        args.gif =args.gif+'_{0}'.format(args.agent_id)

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    print('episode {0}'.format(episode))
    env.seed(2150)
    obs = env.reset()
    total_eps_diversity = 0
    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        RND_observation = torch.tensor(obs['image'], device = device).transpose(0, 2).transpose(1, 2).unsqueeze(0).float()
        state_rep_rew_gen =  torch.flatten(RND_observation, start_dim=1).cpu().numpy()/10
        #get episodic diversity
        eps_div = episodic_buffer.compute_episodic_intrinsic_reward(state_rep_rew_gen)
        total_eps_diversity += eps_div
        episodic_buffer.add_state(state_rep_rew_gen)
        episodic_buffer.compute_new_average()
        print('episodic_diversity_{0}'.format(eps_div))
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)
        print(action)
        if done or env.window.closed:
            print('total_episodic_diversity_reward_{0}'.format(total_eps_diversity/100))
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
