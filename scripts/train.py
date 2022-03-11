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
import os
import numpy as np
from torch_ac.utils import DictList, ParallelEnv
#disable torch debugs for extra speed
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    txt_logger.info("{}\n".format(acmodel))
    acmodels_list.append(acmodel)


#initailize rew_gen, state representer and episodic buffer
rew_gen_list = []
RND_list = []
best_trajectories_list = []
evo_updates = 0
for i in range(0, args.outer_workers):
    utils.seed(args.seed)
    rew_gen = RewGenNet(512, device)
    RND_model = RNDModelNet(device)
    rew_gen_list.append(rew_gen)
    RND_list.append(RND_model)
#initialise master rew gen and master RND
master_rew_gen = RewGenNet(512, device)
master_RND_model = RNDModelNet(device)
master_rew_gen_original = copy.deepcopy(master_rew_gen.state_dict())

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
    else:
        rew_gen_list[i].network_noise = copy.deepcopy(-rew_gen_list[int(i-args.outer_workers/2)].network_noise)
        rew_gen_list[i].update_weights(copy.deepcopy(-rew_gen_list[int(i-args.outer_workers/2)].network_noise))
#copy one policy for all inner agents
agent_to_copy = 0
for i in range(0,args.outer_workers):
    utils.seed(args.seed)
    if i != agent_to_copy:
#        rew_gen_list[i].load_state_dict(copy.deepcopy(rew_gen_list[agent_to_copy].state_dict()))
        RND_list[i].load_state_dict(copy.deepcopy(RND_list[agent_to_copy].state_dict()))
        #acmodels_list[i].load_state_dict(copy.deepcopy(acmodels_list[agent_to_copy].state_dict()))
episodic_buffer = Episodic_buffer()
#save inital random state of each policy agent
policy_agent_params_list = []
for i in range(0, args.outer_workers):
    policy_agent_params_list.append(copy.deepcopy(acmodels_list[i].state_dict()))



# Load algo
algos_list = []
for i in range(0,args.outer_workers):
    utils.seed(args.seed)
    print(i)
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

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
evol_start_time = time.time()
while num_frames < args.frames:
    for i in range(0,args.outer_workers):
        #fix seeds for all the agents exploration
        utils.seed(args.seed)
        update_start_time = time.time()
        exps, logs1 = algos_list[i].collect_experiences()
        logs2 = algos_list[i].update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
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
    #add dummy trajectories to trajectory buffer at the beginning
    if update == 0:
        trajectories_list = []
        for ii in range(0,args.outer_workers):
            evaluator = eval(args.env, algos_list[ii].acmodel.state_dict(), algos_list[ii].RND_model, algos_list[ii].rew_gen_model.state_dict(), ii, argmax = True)
            trajectory, _, _ = evaluator.run()
            trajectories_list.append(trajectory.cpu().numpy())
        sample_number = args.random_samples
        for j in range(0,sample_number):
            random_agent_id = torch.randint(0,args.outer_workers,(1,1)).item()
            random_trajectory = trajectories_list[random_agent_id]
            dummy_trajectory = np.zeros_like(random_trajectory)+100
            episodic_buffer.add_state(dummy_trajectory)
        #compute averge to stop first solution exploding
        #for ii in range(0,args.outer_workers):
        #    episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])
        #episodic_buffer.compute_new_average()
        print('random trajectories added')
    num_frames += logs["num_frames"]
    update += 1
    #do evolutionary update
    if  update % args.updates_per_evo_update == 0:
        #set new random seed for evo updates
        print(num_frames)
        utils.seed(args.seed*542+num_frames-update)
        #eval interactions with env
        #collect trajectories
        trajectories_list = []
        entropy_list = [] 
        episodic_diversity_list = []
        for ii in range(0,args.outer_workers):
            evaluator = eval(args.env, algos_list[ii].acmodel.state_dict(), algos_list[ii].RND_model, algos_list[ii].rew_gen_model.state_dict(), ii, argmax = True)
            trajectory, episodic_diversity, repeatability_factor = evaluator.run()
            trajectories_list.append(trajectory.cpu().numpy())
            #normalize diversity with number of steps
            episodic_diversity_list.append(episodic_diversity/100*repeatability_factor)
        #compute diversity for each outer worker
        print('episodic diversity')
        print(episodic_diversity_list)
        diversity_eval_list = []
        #divide by 10000 to normalize
        for ii in range(0,args.outer_workers):
            diversity = episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])/100
            diversity = min(max(diversity,0.1),10)
            diversity_eval_list.append(diversity)
        #episodic_buffer.compute_new_average()
        print('diversity eval')
        print(diversity_eval_list)
        rollout_eps_diversity = torch.tensor(episodic_diversity_list)
        rollout_diversity_eval = torch.tensor(diversity_eval_list)
        rollout_diversity_eval = rollout_diversity_eval * rollout_eps_diversity
        print(rollout_diversity_eval)
        diversity_ranking = compute_ranking(rollout_diversity_eval,args.outer_workers).to(device)
        #combine noise
        noise_tuple = tuple([algo.rew_gen_model.network_noise for algo in algos_list])
        total_noise = torch.cat(noise_tuple, dim = 0)
        noise_effect_sum = torch.einsum('i j, i -> j',total_noise, diversity_ranking.squeeze())
        best_agent_index = torch.argmax(rollout_diversity_eval)
        rew_gen_weight_updates = args.rew_gen_lr*noise_effect_sum #args.rew_gen_lr*(1/(args.outer_workers*args.noise_std))*noise_effect_sum #args.rew_gen_lr*total_noise[best_agent_index,:].squeeze() #args.rew_gen_lr*(1/(args.outer_workers*args.noise_std))*noise_effect_sum  #total_noise[best_agent_index,:].squeeze() #args.rew_gen_lr*1/(args.outer_workers*args.noise_std)*noise_effect_sum
        best_agent_index = torch.argmax(rollout_diversity_eval)
        #for ii in range(0,args.outer_workers):
        #    if ii != best_agent_index:
        #        rew_gen_list[ii].network_noise = rew_gen_list[best_agent_index].network_noise
        #save most diverse agent
        status = {"num_frames": num_frames, "update": update,
                    "model_state": algos_list[best_agent_index].acmodel.state_dict(), "optimizer_state": algos_list[best_agent_index].optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir, i, best = True, update = update)
        top_trajectories_indexes = torch.topk(rollout_diversity_eval,args.top_trajectories)[1]
        print(top_trajectories_indexes.shape)
        # add trajectories to buffer
        for ii in range(0,top_trajectories_indexes.shape[0]):
            index = int(top_trajectories_indexes[ii].item())
            best_trajectories_list.append(trajectories_list[index])
        if evo_updates % args.trajectory_updates_per_evo_updates == 0 and evo_updates != 0:
            txt_logger.info('diversity buffer updated in evo {0}'.format(evo_updates))
            for item in best_trajectories_list:
                episodic_buffer.add_state(item.squeeze())
            best_trajectories_list = []
        # Train model
        ##update weights in rew_gen master
        #master_rew_gen = RewGenNet(512, device)
        #master_rew_gen.load_state_dict(copy.deepcopy(master_rew_gen_original))
        #print(rew_gen_weight_updates)
        master_rew_gen.update_weights(rew_gen_weight_updates)
        #print(master_rew_gen.state_dict())
        #update weights of each rew gen with master weights and initialize new noise
        for ii in range(0,args.outer_workers):
            algos_list[ii].rew_gen_model.load_state_dict(copy.deepcopy(master_rew_gen.state_dict()))
            #algos_list[ii].rew_gen_model.randomly_mutate(args.noise_std)
            if ii < args.outer_workers/2:
                rew_gen_list[ii].randomly_mutate(args.noise_std, args.outer_workers)
            else:
                rew_gen_list[ii].network_noise = copy.deepcopy(-rew_gen_list[int(ii-args.outer_workers/2)].network_noise)
                rew_gen_list[ii].update_weights(copy.deepcopy(-rew_gen_list[int(ii-args.outer_workers/2)].network_noise))
        evo_updates += 1
        txt_logger.info('evolutionary update complete')
        evol_end_time = time.time()
        txt_logger.info("computation_time_{0}".format(evol_end_time-evol_start_time))
        #write to log
        #convert to floatin point
        rollout_diversity_eval = 1.0*rollout_diversity_eval
        diversity_mean =torch.mean(rollout_diversity_eval)
        diversity_max = torch.max(rollout_diversity_eval)
        diversity_min = torch.min(rollout_diversity_eval)
        diversity_std = torch.std(rollout_diversity_eval)
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
        tb_writer.add_scalar('diversity/mean',diversity_mean, num_frames)  
        tb_writer.add_scalar('diversity/max',diversity_max, num_frames)  
        tb_writer.add_scalar('diversity/min',diversity_min, num_frames)  
        tb_writer.add_scalar('diversity/std',diversity_std, num_frames)       
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
            print(ii)
            print(device)
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
                
        


         
