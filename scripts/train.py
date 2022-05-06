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
from rew_gen.rew_gen_utils import two_point_adaptation

import utils
from utils import device
from scripts import visualize_debug
from scripts.eval import eval
from model import ACModel
import time
import torch
import copy
import time
import os
import numpy as np
from torch_ac.utils import DictList, ParallelEnv
#disable torch debugs for extra speed
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random


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
parser.add_argument("--random_samples", type=int, default=6,
                    help="number of initial random samples for trajectory buffer. Must be lower than number of outer workers")
parser.add_argument("--top_trajectories", type=int, default=2,
                    help="number of top trajectories added to buffer per rew_gen_update. Note top trajectoeries are only added to the knn every trajectory_updates_per_evo_updates")
parser.add_argument("--TPA_agents", type=int, default=2,
                    help="numbers of agent for TPA step update. Note 2 only is currently supported. PArameter for future extension")
parser.add_argument("--alpha", type=float, default=0.4,
                    help="step size decrease factor. Note must be smaller than 1, otherwise step size diverges")
parser.add_argument("--beta", type=float, default=1/0.4,
                    help="step size increase factor (nromally set as 1/alpha, compute manually)")
parser.add_argument("--d_sigma", type=float, default=1,
                    help="TPA sigma update normalization")
parser.add_argument("--c_z", type=float, default=0.5,
                    help="TPA running average coeffecient")


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

#make log for each agent
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
for ii in range(0,args.outer_workers):
    
    utils.seed(args.seed)
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs_list.append(envs)
    txt_logger.info("Environments loaded\n")

# create TPA environments

envs_list_TPA = []
for ii in range(0,args.TPA_agents):
    
    utils.seed(args.seed)
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs_list_TPA.append(envs)
    txt_logger.info("TPA Environments loaded\n")


# Load training status

#try:
#    status = utils.get_status(model_dir)
#except OSError:
status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs_list[0][0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

#load TPA preprocessor

obs_space_TPA, preprocess_obss_TPA = utils.get_obss_preprocessor(envs_list_TPA[0][0].observation_space)
if "vocab" in status:
    preprocess_obss_TPA.vocab.load_vocab(status["vocab"])
txt_logger.info("TPA Observations preprocessor loaded")
action_space_TPA = envs_list_TPA[0][0].action_space

#parallerized TPA envs:
for i in range(0, args.TPA_agents):
    envs_list_TPA[i] = ParallelEnv(envs_list_TPA[i])

action_space = envs_list[0][0].action_space
# Load model
acmodels_list = []
for i in range(0,args.outer_workers):
    utils.seed(args.seed)
    acmodel = ACModel(obs_space, action_space, args.mem, args.text)
    #if "model_state" in status:
    #    acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model {0} loaded\n".format(i))
    #txt_logger.info("{}\n".format(acmodel))
    acmodels_list.append(acmodel)


#initailize rew_gen, state representer, episodic buffer and TPA averaging
rew_gen_list = []
RND_list = []
best_trajectories_list = []
eval_states_list = []
lifetime_returns = torch.zeros(args.outer_workers)
evo_updates = 0
for i in range(0, args.outer_workers):
    utils.seed(args.seed)
    rew_gen = RewGenNet(507, device)
    RND_model = RNDModelNet(device)
    rew_gen_list.append(rew_gen)
    RND_list.append(RND_model)
#initialise master rew gen and master RND
master_rew_gen = RewGenNet(507, device)
master_RND_model = RNDModelNet(device)
master_rew_gen_original = copy.deepcopy(master_rew_gen.state_dict())
network_noise_std = args.noise_std
z = args.rew_gen_lr
rerun_needed = []
master_ACModel_model = ACModel(obs_space, action_space, args.mem, args.text)
##load parameters of just one agent
for i in range(0,args.outer_workers):
        rew_gen_list[i].load_state_dict(copy.deepcopy(master_rew_gen.state_dict()))
        RND_list[i].load_state_dict(copy.deepcopy(master_RND_model.state_dict()))
        acmodels_list[i].load_state_dict(copy.deepcopy(master_ACModel_model.state_dict()))
#initalize noise
for i in range(0,args.outer_workers):
    if i < args.outer_workers/2:
        rew_gen_list[i].randomly_mutate(args.noise_std, args.outer_workers)
        rew_gen_list[i].update_weights(rew_gen_list[i].network_noise)
    else:
        rew_gen_list[i].network_noise = copy.deepcopy(-rew_gen_list[int(i-args.outer_workers/2)].network_noise)
        rew_gen_list[i].update_weights(copy.deepcopy(-rew_gen_list[int(i-args.outer_workers/2)].network_noise))
    rerun_needed.append(True)
#copy one policy for all inner agents
agent_to_copy = 0
for i in range(0,args.outer_workers):
    utils.seed(args.seed)
    if i != agent_to_copy:
#        rew_gen_list[i].load_state_dict(copy.deepcopy(rew_gen_list[agent_to_copy].state_dict()))
        RND_list[i].load_state_dict(copy.deepcopy(RND_list[agent_to_copy].state_dict()))
        #acmodels_list[i].load_state_dict(copy.deepcopy(acmodels_list[agent_to_copy].state_dict()))
episodic_buffer = Episodic_buffer(n_neighbors=10, mu = 0.9, zeta = 50, epsilon = 0.0001, const = 0.001, s_m = 0.04)
#save inital random state of each policy agent
policy_agent_params_list = []
for i in range(0, args.outer_workers):
    policy_agent_params_list.append(copy.deepcopy(acmodels_list[i].state_dict()))



# Load algo
algos_list = []
for i in range(0,args.outer_workers):
    utils.seed(args.seed)
    #parallelrize the envs:
    envs_list[i] = ParallelEnv(envs_list[i])
    if args.algo == "a2c":
        algos_list.append(torch_ac.A2CAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss))
    elif args.algo == "ppo":
        algos_list.append(torch_ac.PPOAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, agent_id = i))
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algos_list[i].optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
# Train model
print(master_ACModel_model)
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
evol_start_time = time.time()
while num_frames < args.frames:
    for i in range(0,args.outer_workers):
        if rerun_needed[i]:
            #fix seeds for all the agents exploration
            utils.seed(args.seed)
            update_start_time = time.time()
            exps, logs1 = algos_list[i].collect_experiences()
            logs2 = algos_list[i].update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()
            lifetime_returns[i] += sum(logs["return_per_episode"])
            #delete after update
            exps = None
            logs1 = None
            #only upodate frames for last agent
            #if i == args.outer_workers-1:
            #    num_frames += logs["num_frames"]
            #    update += 1
        
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
                intrinsic_reward_per_episode = utils.synthesize(logs["intrinsic_reward_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["int_rreturn_" + key for key in intrinsic_reward_per_episode.keys()]
                data += intrinsic_reward_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                header += ['agent_id']
                data += [i]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | id {}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()
    
                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalars(field, {'agent_id_{0}'.format(i):value}, num_frames)   

                if args.save_interval > 0 and update % args.save_interval == 0:
                    status = {"num_frames": num_frames, "update": update,
                            "model_state": algos_list[i].acmodel.state_dict(), "optimizer_state": algos_list[i].optimizer.state_dict()}
                    if hasattr(preprocess_obss, "vocab"):
                        status["vocab"] = preprocess_obss.vocab.vocab
                    utils.save_status(status, model_dir,i)
                    txt_logger.info("Status saved")
        else:
            lifetime_returns[i] = old_lifetime_returns_list[i]
    #add dummy trajectories to trajectory buffer at the beginning
    if update == 0:
        trajectories_list = []
        for ii in range(0,args.outer_workers):
            evaluator = eval(args.env, algos_list[ii].acmodel.state_dict(), algos_list[ii].RND_model, algos_list[ii].rew_gen_model.state_dict(), ii, argmax = True)
            trajectory, _, _, _, _ = evaluator.run()
            trajectories_list.append(trajectory.cpu().numpy())
        sample_number = args.random_samples
        for j in range(0,sample_number):
            random_agent_id = torch.randint(0,args.outer_workers,(1,1)).item()
            random_trajectory = trajectories_list[random_agent_id]
            dummy_trajectory = np.zeros_like(random_trajectory)+100
            episodic_buffer.add_state(random_trajectory)
        #compute averge to stop first solution exploding
        for ii in range(0,args.outer_workers):
            episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])
        episodic_buffer.compute_new_average()
        txt_logger.info('random trajectories added')
    num_frames += logs["num_frames"]
    update += 1
    #do evolutionary update
    if  update % args.updates_per_evo_update == 0:
        #set new random seed for evo updates
        utils.seed(args.seed*542+num_frames-evo_updates)
        #eval interactions with env
        #collect trajectories and compute episodic diversity
        trajectories_list = []
        entropy_list = [] 
        episodic_diversity_list = []
        global_diversity_list = []
        local_states_list = []
        local_states_archive_list  = []
        for ii in range(0,args.outer_workers):
            if rerun_needed[ii]:
                evaluator = eval(args.env, algos_list[ii].acmodel.state_dict(), master_RND_model, algos_list[ii].rew_gen_model.state_dict(), ii, argmax = True)
                #if evo_updates == 20 and ii == args.outer_workers - 2 :
                #    txt_logger.info('cheating enabled')
                #    trajectory, episodic_diversity, repeatability_factor, states_list, lifetime_diversity = evaluator.run(cheat = True)
                #else: 
                trajectory, episodic_diversity, repeatability_factor, states_list, lifetime_diversity = evaluator.run()
                eval_states_list.extend(states_list)
                local_states_list.extend(states_list)
                local_states_archive_list.append(states_list)
                trajectories_list.append(trajectory.cpu().numpy())
                #normalize diversity with number of steps
                episodic_diversity_list.append(episodic_diversity*repeatability_factor)
                lifetime_diversity = min(max(lifetime_diversity,1),10)
                global_diversity_list.append(lifetime_diversity)
            else:
                trajectories_list.append(old_trajectory_list[ii])
                episodic_diversity_list.append(old_episodic_diversity_list[ii])
                global_diversity_list.append(old_global_diversity_list[ii])
                local_states_list.extend(old_local_states_list[ii])
                eval_states_list.extend(old_local_states_list[ii])
                local_states_archive_list.append(old_local_states_list[ii])
        txt_logger.info('episodic diversity')
        txt_logger.info(episodic_diversity_list)
        txt_logger.info('diversity eval')
        txt_logger.info(global_diversity_list)
        rollout_eps_diversity = torch.tensor(episodic_diversity_list)
        rollout_global_diversity = torch.tensor(global_diversity_list)
        rollout_global_diversity_raw = copy.deepcopy(rollout_global_diversity)
        #rollout_global_diversity = compute_ranking(rollout_global_diversity,args.outer_workers)
        lifetime_returns_original = copy.deepcopy(lifetime_returns)
        lifetime_returns = 30*lifetime_returns
        #rollout_global_diversity[rollout_eps_diversity == 0] = 0
        #rollout_eps_diversity[rollout_eps_diversity<=0.001] = rollout_eps_diversity[rollout_eps_diversity<=0.001]*0.001
        rollout_diversity_eval = (rollout_global_diversity * rollout_eps_diversity) + lifetime_returns
        txt_logger.info('overall diversity')
        txt_logger.info(rollout_diversity_eval)
        txt_logger.info('lifetime reward')
        txt_logger.info(lifetime_returns)
        diversity_ranking = compute_ranking(rollout_diversity_eval,args.outer_workers).to(device)
        #combine noise
        noise_tuple = tuple([algo.rew_gen_model.network_noise for algo in algos_list])
        total_noise = torch.cat(noise_tuple, dim = 0)
        noise_effect_sum = torch.einsum('i j, i -> j',total_noise, diversity_ranking.squeeze())
        best_agent_index = torch.argmax(rollout_diversity_eval)
        #if evo_updates == 0:
        #    args.rew_gen_lr = 0.5
        #else:
        #    args.rew_gen_lr = 0.01
        #determine best step size
        for env in envs_list_TPA:
            env.reset()
        new_rew_gen_lr, z = two_point_adaptation(noise_effect_sum, args, master_rew_gen.state_dict(), master_ACModel_model.state_dict(), master_RND_model.state_dict(), episodic_buffer, txt_logger, z, envs_list_TPA, obs_space_TPA, preprocess_obss_TPA, action_space_TPA, master_RND_model.var_mean_reward.mean, master_RND_model.var_mean_reward.var)
        args.rew_gen_lr = new_rew_gen_lr
        utils.seed(args.seed*542+num_frames-evo_updates)
        #utils.seed(args.seed*542+num_frames-evo_updates)
        txt_logger.info('new step size {0}'.format(args.rew_gen_lr))
        rew_gen_weight_updates = args.rew_gen_lr*noise_effect_sum #args.rew_gen_lr*(1/(args.outer_workers*args.noise_std))*noise_effect_sum #args.rew_gen_lr*total_noise[best_agent_index,:].squeeze() #args.rew_gen_lr*(1/(args.outer_workers*args.noise_std))*noise_effect_sum  #total_noise[best_agent_index,:].squeeze() #args.rew_gen_lr*1/(args.outer_workers*args.noise_std)*noise_effect_sum
        best_agent_index = torch.argmax(rollout_diversity_eval)
        #save most diverse agent
        status = {"num_frames": num_frames, "update": update,
                    "model_state": algos_list[best_agent_index].acmodel.state_dict(), "optimizer_state": algos_list[best_agent_index].optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir, i, best = True, update = update)
        top_trajectories_indexes = torch.topk(rollout_diversity_eval,args.top_trajectories)[1]
            #episodic_buffer.moving_average_distance = 0
        # Train model
        ##update weights in rew_gen master
        #master_rew_gen = RewGenNet(512, device)
        #master_rew_gen.load_state_dict(copy.deepcopy(master_rew_gen_original))
        #print(rew_gen_weight_updates)
        #compute distributions ratio
        old_mean = copy.deepcopy(master_rew_gen.get_vectorized_param()).cuda()
        master_rew_gen.update_weights(rew_gen_weight_updates)
        #weight decay to prevent convergence
        #if evo_updates % args.trajectory_updates_per_evo_updates == 0 and evo_updates != 0:
        #weight_norm = torch.linalg.vector_norm(torch.abs(master_rew_gen.get_vectorized_param()))
        #if weight_norm < 0:
        #    print('weight norm negative, check')
        #txt_logger.info(weight_norm)
        #sign_before = torch.sign(master_rew_gen.get_vectorized_param())
        #decayed_weights = master_rew_gen.get_vectorized_param() - torch.sign(master_rew_gen.get_vectorized_param())*0.002*weight_norm
        weights_decay = -master_rew_gen.get_vectorized_param()*0.05
        #sign_after = torch.sign(decayed_weights)
        #decayed_weights[sign_before != sign_after] = 0.0
        #weights_decay_update = decayed_weights - master_rew_gen.get_vectorized_param()
        master_rew_gen.update_weights(weights_decay)
         #reuse old best agent
        rew_gen_list[args.outer_workers-1].load_state_dict(copy.deepcopy(rew_gen_list[best_agent_index].state_dict()))
        rew_gen_list[args.outer_workers-1].network_noise = (rew_gen_list[args.outer_workers-1].get_vectorized_param() - master_rew_gen.get_vectorized_param()).unsqueeze(0)
        new_mean = copy.deepcopy(master_rew_gen.get_vectorized_param()).cuda()
        done = False
        noise_list = []
        rerun_needed = []
        old_trajectory_list = []
        old_episodic_diversity_list = []
        old_lifetime_returns_list = []
        old_global_diversity_list = []
        old_local_states_list = []
        sampling_number = 0
        while not done:
            sample_flag_1 = random.uniform(0,1)
            sample_flag_2 = random.uniform(0,1)
            sample_agent = random.randint(0, args.outer_workers-1)
            old_parameters =  copy.deepcopy(rew_gen_list[sample_agent].network_noise.squeeze()).cuda() + old_mean
           # coz they are all identity it is uncessary to use identity matrix
            covariance_matrix_inverse = 1/args.noise_std #* torch.eye(new_mean.shape[0]).cpu()
            #new_calculation = torch.matmul((new_mean - old_parameters), (new_mean - old_parameters))
            #old_calculation = torch.matmul((old_mean - old_parameters), (old_mean - old_parameters))
            new_calculation_2 = torch.matmul((old_parameters - new_mean), ( old_parameters- new_mean))
            old_calculation_2 = torch.matmul((old_parameters - old_mean ), ( old_parameters - old_mean))
            #distributions_ratio_old = torch.exp(0.5*covariance_matrix_inverse*(new_calculation-old_calculation))
            distributions_ratio_old_2 = torch.exp(-0.5*covariance_matrix_inverse*(new_calculation_2-old_calculation_2))
            #if min(1, distributions_ratio_old_2.item()) > sample_flag_1:
            #    txt_logger.info('reuse old agent')
            #    txt_logger.info(' sampling number {0}'.format(sampling_number))
            #    txt_logger.info(distributions_ratio_old_2.item())
            #    old_parameters = (rew_gen_list[sample_agent].get_vectorized_param() - master_rew_gen.get_vectorized_param())
            #    noise_list.append(old_parameters)
            #    rerun_needed.append(False)
            #    # for all agents not requiring rerun save
            #    old_trajectory_list.append(trajectories_list[sample_agent])
            #    old_episodic_diversity_list.append(episodic_diversity_list[sample_agent])
            #    old_lifetime_returns_list.append(lifetime_returns_original[sample_agent])
            #    old_global_diversity_list.append(global_diversity_list[sample_agent])
            rew_gen_list[sample_agent].randomly_mutate(network_noise_std, args.outer_workers)
            new_parameters = rew_gen_list[sample_agent].network_noise.squeeze() + new_mean
            #new_calculation = torch.matmul((new_mean - new_parameters), (new_mean - new_parameters))
            #old_calculation = torch.matmul((old_mean - new_parameters), (old_mean - new_parameters))
            #distributions_ratio_new = torch.exp(0.5*covariance_matrix_inverse*(old_calculation-new_calculation))
            new_calculation_2 = torch.matmul((new_parameters - new_mean), ( new_parameters- new_mean))
            old_calculation_2 = torch.matmul((new_parameters - old_mean ), ( new_parameters - old_mean))
            distributions_ratio_new_2 = torch.exp(-0.5*covariance_matrix_inverse*(old_calculation_2-new_calculation_2))
            #print(covariance_matrix_inverse)
            #print(new_calculation_2)
            #print(old_calculation_2)
            #print(old_calculation_2-new_calculation_2)
            #print(distributions_ratio_new_2)
            #exit()
            txt_logger.info('noise is {0}'.format(network_noise_std))
            if True: #max(0,1-distributions_ratio_new_2.item()) > sample_flag_2 or sampling_number > 10000:
                txt_logger.info('create new')
                txt_logger.info(' sampling number {0}'.format(sampling_number))
                noise_list.append(new_parameters-new_mean)
                noise_list.append(-(new_parameters-new_mean))
                txt_logger.info(distributions_ratio_new_2.item())
                rerun_needed.append(True)
                rerun_needed.append(True)
                old_trajectory_list.append(None)
                old_trajectory_list.append(None)
                old_episodic_diversity_list.append(None)
                old_episodic_diversity_list.append(None)
                old_lifetime_returns_list.append(None)
                old_lifetime_returns_list.append(None)
                old_global_diversity_list.append(None)
                old_global_diversity_list.append(None)
                old_local_states_list.append(None)
                old_local_states_list.append(None)
            if len(noise_list) >= args.outer_workers-1:
                if len(noise_list) > args.outer_workers-1:
                    for i in range(0, len(noise_list) - args.outer_workers-1):
                        noise_list.pop()
                        rerun_needed.pop()
                        old_trajectory_list.pop()
                        old_episodic_diversity_list.pop()
                        old_lifetime_returns_list.pop()
                        old_global_diversity_list.pop()

                #last agent reuses old weights by default
                rerun_needed.append(False)
                old_trajectory_list.append(trajectories_list[best_agent_index])
                old_episodic_diversity_list.append(episodic_diversity_list[best_agent_index])
                old_lifetime_returns_list.append(lifetime_returns_original[best_agent_index])
                old_global_diversity_list.append(global_diversity_list[best_agent_index])
                print(len(local_states_archive_list))
                print(best_agent_index)
                old_local_states_list.append(local_states_archive_list[best_agent_index])
                break
            sampling_number +=1

                # add trajectories to buffer
        network_noise_std = args.noise_std
        for ii in range(0,top_trajectories_indexes.shape[0]):
            index = int(top_trajectories_indexes[ii].item())
            best_trajectories_list.append(trajectories_list[index])
        #update weights of each rew gen with master weights and initialize new noise
        for ii in range(0,args.outer_workers-1):
            rew_gen_list[ii].load_state_dict(copy.deepcopy(master_rew_gen.state_dict()))
            rew_gen_list[ii].network_noise = copy.deepcopy(noise_list[ii]).unsqueeze(0)
            rew_gen_list[ii].update_weights(rew_gen_list[int(ii)].network_noise)
        #if evo_updates == 20:
        #    exit()
        evo_updates += 1
        txt_logger.info('evolutionary update {0} complete'.format(evo_updates))
        evol_end_time = time.time()
        txt_logger.info("computation_time_{0}".format(evol_end_time-evol_start_time))
        if evo_updates % args.trajectory_updates_per_evo_updates == 0 and evo_updates != 0:
            txt_logger.info('diversity buffer updated in evo {0}'.format(evo_updates))
            master_RND_model.train(eval_states_list)
            master_RND_model.compute_new_mean_and_std(torch.cat(eval_states_list))
            eval_states_list = []
            # when trajectories updated, all agents need to update
            noise_list = []
            rerun_needed = []
            old_trajectory_list = []
            old_episodic_diversity_list = []
            old_lifetime_returns_list = []
            old_global_diversity_list = []
            old_local_states_list = []
            rerun_needed = []
            for ii in range(0, args.outer_workers):
                rerun_needed.append(True)
        #print(master_rew_gen.state_dict())
        #write to log
        #convert to floatin point
        rollout_diversity_eval = 1.0*rollout_diversity_eval
        #report_rollout_diversity_eval = (rollout_global_diversity * rollout_eps_diversity* rollout_global_diversity) + lifetime_returns
        #report_rollout_diversity_eval = ((rollout_global_diversity-lifetime_returns)* rollout_diversity_eval) + lifetime_returns
        report_rollout_diversity_eval = rollout_diversity_eval 
        diversity_mean =torch.mean(report_rollout_diversity_eval)
        diversity_max = torch.max(report_rollout_diversity_eval)
        diversity_min = torch.min(report_rollout_diversity_eval)
        diversity_std = torch.std(report_rollout_diversity_eval)
        mean_list = []
        var_list = []
        agent_layers = master_rew_gen.state_dict()
        for layer in agent_layers:
            mean_layer = agent_layers[layer].mean().cpu().numpy()
            var_layer = np.square(agent_layers[layer].std().cpu().numpy())
            mean_list.append(mean_layer)
            var_list.append(var_layer)
        weight_mean = np.array(mean_list).mean()
        weight_std = np.sqrt(np.nanmean(np.array(var_list)))
        tb_writer.add_scalar('parameters/reward_net_mean_master', weight_mean, num_frames)
        tb_writer.add_scalar('parameters/reward_net_std_master',weight_std, num_frames)
        tb_writer.add_scalar('diversity_total/mean',diversity_mean, num_frames)  
        tb_writer.add_scalar('diversity_total/max',diversity_max, num_frames)  
        tb_writer.add_scalar('diversity_total/min',diversity_min, num_frames)  
        tb_writer.add_scalar('diversity_total/std',diversity_std, num_frames) 

        diversity_eps_mean = torch.mean(rollout_eps_diversity)
        diversity_eps_max = torch.max(rollout_eps_diversity)
        diversity_eps_min = torch.min(rollout_eps_diversity)
        diversity_eps_std = torch.std(rollout_eps_diversity)
        tb_writer.add_scalar('diversity_eps/mean',diversity_eps_mean, num_frames)  
        tb_writer.add_scalar('diversity_eps/max',diversity_eps_max, num_frames)  
        tb_writer.add_scalar('diversity_eps/min',diversity_eps_min, num_frames)  
        tb_writer.add_scalar('diversity_eps/std',diversity_eps_std, num_frames) 

        #report_diversity_global = rollout_global_diversity * rollout_global_diversity_raw*1.0
        report_diversity_global = rollout_global_diversity * 1.0
        diversity_global_mean = torch.mean(report_diversity_global)
        diversity_global_max = torch.max(report_diversity_global)
        diversity_global_min = torch.min(report_diversity_global)
        diversity_global_std = torch.std(report_diversity_global)
        tb_writer.add_scalar('diversity_global/mean',diversity_global_mean, num_frames)  
        tb_writer.add_scalar('diversity_global/max',diversity_global_max, num_frames)  
        tb_writer.add_scalar('diversity_global/min',diversity_global_min, num_frames)  
        tb_writer.add_scalar('diversity_global/std',diversity_global_std, num_frames)

        # report lifetime returns
        lifetime_reward_mean = torch.mean(lifetime_returns)
        lifetime_reward_max = torch.max(lifetime_returns)
        lifetime_reward_min = torch.min(lifetime_returns)
        lifetime_reward_std = torch.std(lifetime_returns)
        tb_writer.add_scalar('lifetime_reward/mean',lifetime_reward_mean, num_frames)  
        tb_writer.add_scalar('lifetime_reward/max', lifetime_reward_max, num_frames)  
        tb_writer.add_scalar('lifetime_reward/min',lifetime_reward_min, num_frames)  
        tb_writer.add_scalar('lifetime_reward/std',lifetime_reward_std, num_frames)


        tb_writer.add_scalar('evo_step_size', args.rew_gen_lr, num_frames)        
        evol_start_time = time.time()
        # reset the inner agents to start a new episode
        #for ii in range(0,args.outer_workers):
        #    utils.seed(args.seed)
        #    print('reset agent {0}'.format(ii))
        #    algos_list[ii].reset()
        # reinitialize the ACM models fully
        #close the env agents, otherwise they will hang in memory forever
        for ii in range(0,args.outer_workers):
            algos_list[ii].reset()
            #algos_list[ii].env.close()
        acmodels_list.clear()
        algos_list.clear()
        algos_list = []
        acmodels_list = []
        lifetime_returns = torch.zeros(args.outer_workers)
        #create new networks
        #obs_space, preprocess_obss = utils.get_obss_preprocessor(envs_list[0][0].observation_space)
        for ii in range(0,args.outer_workers):
            utils.seed(args.seed)
            acmodel = ACModel(obs_space, action_space, args.mem, args.text)
            #if "model_state" in status:
            #acmodel.load_state_dict(status["model_state"])
            acmodel.to(device)
            acmodel.load_state_dict(copy.deepcopy(master_ACModel_model.state_dict()))
            acmodels_list.append(acmodel)
        for ii in range(0,args.outer_workers):
            utils.seed(args.seed)
            envs_list[ii].reset()
            print(rew_gen_list[ii].get_vectorized_param())
            if args.algo == "a2c":
                algos_list.append(torch_ac.A2CAlgo(envs_list[ii], acmodels_list[ii], rew_gen_list[ii], RND_list[ii], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss))
            elif args.algo == "ppo":
                algos_list.append(torch_ac.PPOAlgo(envs_list[ii], acmodels_list[ii], rew_gen_list[ii], RND_list[ii], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, agent_id = ii))
            else:
                raise ValueError("Incorrect algorithm name: {}".format(args.algo))
            txt_logger.info("Optimizer loaded\n")
                
        


         
