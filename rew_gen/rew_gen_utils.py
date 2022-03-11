import torch
import os
import numpy as np
import math

def tensor(x,device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x

def compute_ranking(results_vector,num_workers):
    #add 1 for ranking to start at 1
    ranking_index = torch.tensor(np.argsort(np.argsort(-results_vector.squeeze().cpu().numpy(), kind = 'stable')))+1
    #apply the utility to the rankings
    ranking_numerator = torch.clamp(math.log(num_workers/2+1) - torch.log(ranking_index), min = 0)
    ranking_denominator = torch.sum(torch.clamp(math.log(num_workers/2+1) - torch.log(ranking_index), min = 0))
    ranking = ranking_numerator/ranking_denominator - 1/num_workers
    return ranking