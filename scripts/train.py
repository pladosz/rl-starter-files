import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
from rew_gen.rew_gen_model import RewGenNet
from rew_gen.RND_model import RNDModelNet
from rew_gen.episodic_buffer import Episodic_buffer
from rew_gen.rew_gen_utils import compute_ranking

import utils
from utils import device
from scripts import visualize_debug
from scripts.eval import eval
from model import ACModel
import time
import torch
import copy
import time

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--outer_workers", type=int, default=16,
                    help="number of evolutionary workers")
parser.add_argument("--updates_per_evo_update", type=int, default=60,
                    help="number of training steps of policy agent before rew_gen is updated")
parser.add_argument("--noise_std", type=float, default=0.32,
                    help="noise used for generating new evolutionary agents")
parser.add_argument("--rew_gen_lr", type=float, default = 0.001,
                    help="learning rate of outer evolutionary agent")
parser.add_argument("--trajectory_updates_per_evo_updates", type=int, default=10,
                    help="number of evolutionary steps before trajectories get added to the buffer")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=8,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

args = parser.parse_args()

args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

txt_logger.info(f"Device: {device}\n")

# Load environments
#create multiple envs
envs_list = []
for i in range(0,args.outer_workers):
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs_list.append(envs)
    txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs_list[0][0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
acmodels_list = []
for i in range(0,args.outer_workers):
    acmodel = ACModel(obs_space, envs_list[0][0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model {0} loaded\n".format(i))
    txt_logger.info("{}\n".format(acmodel))
    acmodels_list.append(acmodel)


#initailize rew_gen, state representer and episodic buffer
rew_gen_list = []
RND_list = []
best_trajectories_list = []
evo_updates = 0
for i in range(0, args.outer_workers):
    rew_gen = RewGenNet(512,device)
    RND_model = RNDModelNet(device)
    rew_gen_list.append(rew_gen)
    RND_list.append(RND_model)
#initialise master rew gen and master RND
master_rew_gen = RewGenNet(512,device)
master_RND_model = RNDModelNet(device)
#load parameters of just one agent
for i in range(0,args.outer_workers):
        rew_gen_list[i].load_state_dict(master_rew_gen.state_dict())
        RND_list[i].load_state_dict(master_RND_model.state_dict())
#initalize noise
for i in range(0,args.outer_workers):
    rew_gen_list[i].randomly_mutate(args.noise_std)
episodic_buffer = Episodic_buffer()
#save inital random state of each policy agent
policy_agent_params_list = []
for i in range(0, args.outer_workers):
    policy_agent_params_list.append(copy.deepcopy(acmodels_list[i].state_dict()))

# Load algo
algos_list = []
for i in range(0,args.outer_workers):
    if args.algo == "a2c":
        algos_list.append(torch_ac.A2CAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss))
    elif args.algo == "ppo":
        algos_list.append(torch_ac.PPOAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, agent_id = i))
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algos_list[i].optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
evol_start_time = time.time()
while num_frames < args.frames:
    # Update model parameters

    for i in range(0,args.outer_workers):
        update_start_time = time.time()
        exps, logs1 = algos_list[i].collect_experiences()
        logs2 = algos_list[i].update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        #delete after update
        exps = None
        logs1 = None
        #only upodate frames for 1st agent
        if i == 0:
            num_frames += logs["num_frames"]
            update += 1
        
        #save final animation
        #if update == 79:
        #    visualiser = visualize_debug.eval_visualise(args.env,acmodels_list[i].state_dict(),i, argmax = True)
        #    visualiser.run()
        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
            header += ['agent_id']
            data += [i]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | id {}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                           "model_state": acmodels_list[i].state_dict(), "optimizer_state": algos_list[i].optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir,i)
                txt_logger.info("Status saved")
        #do evolutionary update
        if update % args.updates_per_evo_update == 0 and i == args.outer_workers-1:
            #eval interactions with env
            #collect trajectories
            trajectories_list = []
            entropy_list = []   
            for ii in range(0,args.outer_workers):
                evaluator = eval(args.env,acmodels_list[i].state_dict(), RND_list[i].state_dict(), rew_gen_list[i].state_dict(),i, argmax = True)
                trajectory = evaluator.run()
                trajectories_list.append(trajectory.cpu().numpy())
            #add random trajectories to trajectory buffer
            if update == 0:
                sample_number = 10
                for j in range(0,sample_number):
                    random_agent_id = torch.randint(0,args.outer_workers,(1,1)).item()
                    random_trajectory = trajectories_list[random_agent_id]
                    episodic_buffer.add_state(random_trajectory)
            #compute diversity for each outer worker
            diversity_eval_list = []
            for ii in range(0,args.outer_workers):
                diversity = episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])
                diversity_eval_list.append(diversity)
            episodic_buffer.compute_new_average()
            rollout_diversity_eval = torch.tensor(diversity_eval_list)
            diversity_ranking = compute_ranking(rollout_diversity_eval,args.outer_workers).to(device)
            #combine noise
            #TODO deal with resetting parameters in policy network
            #TODO master rew gen network is necessary to copy parameters from
            noise_tuple = tuple([rew_gen_network.network_noise for rew_gen_network in rew_gen_list])
            total_noise = torch.cat(noise_tuple, dim = 0)
            noise_effect_sum = torch.einsum('i j, i -> j',total_noise, diversity_ranking.squeeze())
            rew_gen_weight_updates = args.rew_gen_lr*1/(args.outer_workers*args.noise_std)*noise_effect_sum
            ##update weights in rew_gen master
            master_rew_gen.update_weights(rew_gen_weight_updates)
            #update weights of each agent with master weights and initialize new noise
            for ii in range(0,args.outer_workers):
                rew_gen_list[ii].load_state_dict(master_rew_gen.state_dict())
                rew_gen_list[ii].randomly_mutate(args.noise_std)
            #deal with policy weights (currently reset the weights to random ones)
            for ii in range(0,args.outer_workers):
                acmodels_list[ii].load_state_dict(policy_agent_params_list[i])
            # add trajectories to buffer
            best_agent_index = torch.argmax(rollout_diversity_eval)
            top_trajectories_indexes = torch.topk(rollout_diversity_eval,2)[1]
            for i in range(0,top_trajectories_indexes.shape[0]):
                index = int(top_trajectories_indexes[i].item())
                best_trajectories_list.append(trajectories_list[index])
            if evo_updates % args.trajectory_updates_per_evo_updates == 0:
                for item in best_trajectories_list:
                    episodic_buffer.add_state(item.squeeze())
                best_trajectories_list = []
            evo_updates += 1
            print('evolutionary update complete')
            evol_end_time = time.time()
            print("computation_time")
            print(evol_end_time-evol_start_time)
            evol_start_time = time.time()

         
