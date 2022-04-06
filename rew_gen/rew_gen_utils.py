from asyncio.log import logger
from re import U
from unicodedata import decimal
import torch
import os
import numpy as np
import math

# imports for two points adaptation

import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
from rew_gen.rew_gen_model import RewGenNet
from rew_gen.RND_model import RNDModelNet
from rew_gen.episodic_buffer import Episodic_buffer
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


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x


def compute_ranking(results_vector, num_workers):
    # add 1 for ranking to start at 1
    ranking_index = torch.tensor(np.argsort(
        np.argsort(-results_vector.squeeze().cpu().numpy(), kind='stable')))+1
    # apply the utility to the rankings
    ranking_numerator = torch.clamp(
        math.log(num_workers/2+1) - torch.log(ranking_index), min=0)
    ranking_denominator = torch.sum(torch.clamp(
        math.log(num_workers/2+1) - torch.log(ranking_index), min=0))
    ranking = ranking_numerator/ranking_denominator - 1/num_workers
    return ranking

def update_weights(update_factor, network, z, args, weights_updates, c_z):
    updated_z = (1-c_z)*z + c_z*math.log(update_factor)
    if updated_z > 2.5:
        updated_z = 2.5
    candidate_rew_gen_lr = args.rew_gen_lr*math.exp(updated_z/args.d_sigma)
    if candidate_rew_gen_lr > 7:
        candidate_rew_gen_lr = 7
    if candidate_rew_gen_lr < 0.05:
        candidate_rew_gen_lr = 0.05
    network.network_noise = copy.deepcopy(
        candidate_rew_gen_lr*weights_updates)
    network.update_weights(
        copy.deepcopy(network.network_noise))
    return updated_z, update_factor


def two_point_adaptation(weights_updates, args, master_weights, acmodel_weights, RND_weights, episodic_buffer, txt_logger, z, envs_list_TPA, obs_space, preprocess_obss, action_space):
    # create multiple envs
    status = {"num_frames": 0, "update": 0}
    envs_list = envs_list_TPA


    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodels_list = []
    for i in range(0, args.TPA_agents):
        utils.seed(args.seed)
        acmodel = ACModel(obs_space, action_space, args.mem, args.text)
        # if "model_state" in status:
        #    acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model {0} loaded\n".format(i))
        #txt_logger.info("{}\n".format(acmodel))
        acmodels_list.append(acmodel)


    # initailize rew_gen, state representer and episodic buffer
    rew_gen_list = []
    RND_list = []
    best_trajectories_list = []
    lifetime_returns = torch.zeros(args.TPA_agents)
    evo_updates = 0
    for i in range(0, args.TPA_agents):
        utils.seed(args.seed)
        rew_gen = RewGenNet(512, device)
        RND_model = RNDModelNet(device)
        rew_gen_list.append(rew_gen)
        RND_list.append(RND_model)
    # initialise master rew gen and master RND
    master_rew_gen = RewGenNet(512, device)
    master_rew_gen.load_state_dict(copy.deepcopy(master_weights))
    master_RND_model = RNDModelNet(device)
    master_RND_model.load_state_dict(copy.deepcopy(RND_weights))
    master_rew_gen_original = copy.deepcopy(master_rew_gen.state_dict())

    master_ACModel_model = ACModel(obs_space, action_space, args.mem, args.text)
    master_ACModel_model.load_state_dict(copy.deepcopy(acmodel_weights))
    #   load parameters of just one agent
    for i in range(0, args.TPA_agents):
        rew_gen_list[i].load_state_dict(copy.deepcopy(master_rew_gen.state_dict()))
        RND_list[i].load_state_dict(
            copy.deepcopy(master_RND_model.state_dict()))
        acmodels_list[i].load_state_dict(
            copy.deepcopy(master_ACModel_model.state_dict()))
    # initalize noise for TPA
    #z = (1-args.c_z)*z + args.c_z*math.log(args.alpha)*int(rollout_diversity_eval[0].item()>rollout_diversity_eval[1].item()) + args.c_z*math.log(args.beta)*int(rollout_diversity_eval[0].item()<=rollout_diversity_eval[1].item())
    #hard limit on step size to prevent network saturation
    #if rollout_diversity_eval[0] == rollout_diveoh this isrsity_eval[1]:
    #    new_rew_gen_lr = args.rew_gen_lr
    #else:
    
    factor_matrix = []
    c_z = args.c_z*torch.ones(args.TPA_agents)
    for i in range(0, args.TPA_agents):
        if i == 0:
            if args.rew_gen_lr < 0.1:
                c_z[i] = 1.5 * c_z[i]
                update_weights(6*args.beta, rew_gen_list[i], z, args, weights_updates,c_z[i].item())
                factor_matrix.append(6*args.beta)
            else:
                update_weights(2*args.beta, rew_gen_list[i], z, args, weights_updates, c_z[i].item())
                factor_matrix.append(2*args.beta)
        #if i == 1:
        #    rew_gen_list[i].network_noise = copy.deepcopy(
        #        weights_updates)
        #    rew_gen_list[i].update_weights(
        #        copy.deepcopy(weights_updates))
        elif i == 1:
            update_weights(args.beta, rew_gen_list[i], z, args, weights_updates, c_z[i].item())
            factor_matrix.append(args.beta)
        elif i == 2:
            update_weights(args.alpha, rew_gen_list[i], z, args, weights_updates, c_z[i].item())
            factor_matrix.append(args.alpha)
        elif i == 3:
            if args.rew_gen_lr > 3.5:
                c_z[i] = 1.9 * c_z[i]
                update_weights(0.01*args.alpha, rew_gen_list[i], z, args, weights_updates, c_z[i].item())
                factor_matrix.append(0.01*args.alpha)
            else:
                update_weights(0.25*args.alpha, rew_gen_list[i], z, args, weights_updates, c_z[i].item())
                factor_matrix.append(0.25*args.alpha)
        else:
            print('you defined extra TPA agents but did not define what to do with them. fix it. Exiting!')
            print(i)
            print(args.TPA_agents)
            exit()
    factor_matrix = torch.tensor(factor_matrix)
    # copy one policy for all inner agents
    agent_to_copy = 0
    for i in range(0, args.TPA_agents):
        utils.seed(args.seed)
        if i != agent_to_copy:
            #        rew_gen_list[i].load_state_dict(copy.deepcopy(rew_gen_list[agent_to_copy].state_dict()))
            RND_list[i].load_state_dict(copy.deepcopy(
                RND_list[agent_to_copy].state_dict()))
            #  acmodels_list[i].load_state_dict(copy.deepcopy(acmodels_list[agent_to_copy].state_dict()))
    # save inital random state of each policy agent
    policy_agent_params_list = []
    for i in range(0, args.TPA_agents):
        policy_agent_params_list.append(
            copy.deepcopy(acmodels_list[i].state_dict()))
    
        # Load algo
    algos_list = []
    for i in range(0, args.TPA_agents):
        utils.seed(args.seed)
        print(i)
        #parallelrize the envs:
        if args.algo == "a2c":
            algos_list.append(torch_ac.A2CAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss, txt_logger = txt_logger))
        elif args.algo == "ppo":
            algos_list.append(torch_ac.PPOAlgo(envs_list[i], acmodels_list[i], rew_gen_list[i], RND_list[i], args.procs, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, agent_id = i))
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))
        txt_logger.info("Optimizer loaded\n")
    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    evol_start_time = time.time()
    while update <= args.updates_per_evo_update:
        for i in range(0, args.TPA_agents):
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
            #if i == args.TPA_agents-1:
            #    num_frames += logs["num_frames"]
            #    update += 1
        
            #save final animation
            #if update == 79:
            #    visualiser = visualize_debug.eval_visualise(args.env,acmodels_list[i].state_dict(),i, argmax = True)
            #    visualiser.run()
            # Print logs
            if update % args.log_interval == 0:
                txt_logger.info('TPA logs')
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

        #compute averge to stop first solution exploding
        #for ii in range(0,args.TPA_agents):
        #    episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])
        #episodic_buffer.compute_new_average()
        num_frames += logs["num_frames"]
        update += 1
    #evaluate the solutions
    #set new random seed for evo updates
    utils.seed(args.seed*542+num_frames-evo_updates)
    #eval interactions with env
    #collect trajectories
    trajectories_list = []
    entropy_list = [] 
    episodic_diversity_list = []
    for ii in range(0,args.TPA_agents):
        evaluator = eval(args.env, algos_list[ii].acmodel.state_dict(), algos_list[ii].RND_model, algos_list[ii].rew_gen_model.state_dict(), ii, argmax = True)
        trajectory, episodic_diversity, repeatability_factor = evaluator.run()
        trajectories_list.append(trajectory.cpu().numpy())
        #normalize diversity with number of steps
        episodic_diversity_list.append(episodic_diversity/100*repeatability_factor)
    #compute diversity for each outer worker
    txt_logger.info('episodic diversity TPA')
    txt_logger.info(episodic_diversity_list)
    global_diversity_list = []
    #divide by 10000 to normalize
    for ii in range(0,args.TPA_agents):
        diversity = episodic_buffer.compute_episodic_intrinsic_reward(trajectories_list[ii])/100
        diversity = min(max(diversity,0.1),10)
        global_diversity_list.append(diversity)
    #episodic_buffer.compute_new_average()
    txt_logger.info('global diversity')
    txt_logger.info(global_diversity_list)
    rollout_eps_diversity = torch.tensor(episodic_diversity_list)
    rollout_global_diversity = torch.tensor(global_diversity_list)
    lifetime_returns = 10*lifetime_returns
    rollout_diversity_eval = (rollout_global_diversity * rollout_eps_diversity) + lifetime_returns
    txt_logger.info('diversity eval TPA')
    #compute step update
    best_agent = torch.argmax(rollout_diversity_eval)
    step_update_factor = factor_matrix[best_agent].item()
    c_z = c_z[best_agent].item()
    txt_logger.info(step_update_factor)
    rollout_diversity_eval = rollout_diversity_eval.numpy()
    rollout_diversity_eval = np.round(rollout_diversity_eval, 5)
    txt_logger.info(rollout_diversity_eval)
    _,count = np.unique(rollout_diversity_eval, return_counts = True)
    txt_logger.info(rollout_diversity_eval.shape)
    txt_logger.info(count)

    if np.max(count) == rollout_diversity_eval.shape[0]:
        txt_logger.info('all outputs equal!')
        step_update_factor = torch.max(factor_matrix).item()
        c_z = args.c_z
    #txt_logger.info(int(rollout_diversity_eval[0].item()<rollout_diversity_eval[1].item()))
    #txt_logger.info(int(rollout_diversity_eval[0].item()>=rollout_diversity_eval[1].item()))
    z = (1-c_z)*z + c_z*math.log(step_update_factor)
    #z = (1-args.c_z)*z + args.c_z*math.log(args.alpha)*int(rollout_diversity_eval[0].item()>rollout_diversity_eval[1].item()) + args.c_z*math.log(args.beta)*int(rollout_diversity_eval[0].item()<=rollout_diversity_eval[1].item())
    #hard limit on step size to prevent network saturation
    if z > 2.5:
        z = 2.5
    txt_logger.info('z')
    txt_logger.info(z)
    #if rollout_diversity_eval[0] == rollout_diveoh this isrsity_eval[1]:
    #    new_rew_gen_lr = args.rew_gen_lr
    #else:
    new_rew_gen_lr = args.rew_gen_lr*math.exp(z/args.d_sigma)
    if new_rew_gen_lr > 7:
        new_rew_gen_lr = 7
    if new_rew_gen_lr < 0.05:
        new_rew_gen_lr = 0.05
    # clear models
    for ii in range(0,args.TPA_agents):
        algos_list[ii].reset()
    acmodels_list.clear()
    algos_list.clear()
    algos_list = []
    acmodels_list = []
    lifetime_returns = torch.zeros(args.TPA_agents)
    txt_logger.info('TPA updated!')


    return new_rew_gen_lr , z
