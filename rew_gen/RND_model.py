import torch
import numpy as np
from torch._six import inf

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RNDModelNet(torch.nn.Module):
    #code taken from https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/model.py
    def __init__(self,device, epsilon=1e-4, num_workers= 16, learning_rate = 1e-4, epoch = 8, update_proportion = 0.5, batch_size = 512):
        super(RNDModelNet, self).__init__()
        self.device = device
        feature_output = 6400
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(feature_output, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

        self.target = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(feature_output, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

        for p in self.modules():
            if isinstance(p, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, torch.nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False
        self.to(self.device)
        #initalize running mean/var counters for reward and images
        #self.var_mean_reward = RunningMeanStd(epsilon = epsilon)
        #self.var_mean_obs = RunningMeanStd(epsilon = epsilon)
        self.apply(init_params)
        #for now here, maybe creating separate agent is a better idea
        self.optimizer = torch.optim.Adam( list(self.predictor.parameters()),
                                    lr=learning_rate)
        self.epoch = epoch
        self.update_proportion = update_proportion
        self.batch_size = batch_size
        self.var_mean_reward = self.RunningMeanStd(epsilon = epsilon)
        self.var_mean_obs = self.RunningMeanStd(epsilon = epsilon)

    def forward(self, next_obs):
        #next_obs = torch.tensor(next_obs).to(self.device)
        #next_obs = self.compress_frames(next_obs)
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature

    def get_state_rep(self, obs):
        obs = torch.tensor(obs).to(self.device)
        #obs = self.compress_frames(obs)
        with torch.no_grad():
            target_feature = self.target(obs).detach()
        return target_feature

    def extract_frames(self, obs):
        "this function extracts last frame from stack and normalizes it for inputs to the RND"
        obs = obs[:,3,:,:].detach()
        self.var_mean_obs.update(obs.cpu().numpy())
        mean=torch.tensor(self.var_mean_obs.mean).to(self.device)
        std=torch.tensor(self.var_mean_obs.var).to(self.device)
        mean = mean[:, None, None]
        std = std[:, None, None]
        if torch.any(std == 0):
            std[std == 0] = 1
        obs =  (obs - mean).div(std)
        obs = torch.clamp(obs, min = -5, max = 5)
        obs = obs[:,None, :, :]
        return obs

    def rgb2gray(self, rgb):
        #TODO not sure state rep will work fine
        gray_obs = torch.einsum('i j k l, j -> i k l',rgb, torch.tensor([0.2989, 0.5870, 0.1140]).to(self.device))
        #gray_obs = torch.unsqueeze(gray_obs, 1)
        return gray_obs

    def compress_frames(self, obs):
        if len(obs.shape) == 5:
            obs=obs.squeeze(0)
        "this function compresses RGB frames to grey scale"
        obs = obs.detach()
        obs = self.rgb2gray(obs)
        #self.var_mean_obs.update(obs.cpu().numpy())
        #mean=torch.tensor(self.var_mean_obs.mean)
        #std=torch.tensor(self.var_mean_obs.var)
        #mean = mean[None, :, :]
        #std = std[None, :, :]
        #if torch.any(std == 0):
        #    std[std == 0] = 1
        #obs =  (obs - mean).div(std)
        #obs = torch.clamp(obs, min = -5, max = 5)
        obs = obs[:,None, :, :]
        return obs

    def compute_new_mean_and_std(self, next_obs):
        #next_obs = torch.tensor(next_obs).to(self.device).detach()
        target_next_feature = self.target(next_obs).detach()
        predict_next_feature = self.predictor(next_obs).detach()
        intrinsic_reward = ((predict_next_feature - target_next_feature).pow(2).sum(1) / 2).detach().cpu().numpy()
        #intrinsic_reward = torch.dist(predict_next_feature, target_next_feature, 2)
        mean, std, count = np.mean(intrinsic_reward), np.std(intrinsic_reward), len(intrinsic_reward)
        self.var_mean_reward.update_from_moments(mean, std ** 2, count)
    
    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.tensor(next_obs).to(self.device).detach()
        #next_obs = self.compress_frames(next_obs).to(self.device)
        target_next_feature = self.target(next_obs).detach()
        predict_next_feature = self.predictor(next_obs).detach()
        intrinsic_reward = ((predict_next_feature - target_next_feature).pow(2).sum(1) / 2).detach().cpu().numpy()
        #intrinsic_reward = torch.dist(predict_next_feature, target_next_feature, 2)
        #mean, std, count = np.mean(intrinsic_reward), np.std(intrinsic_reward), len(intrinsic_reward)
        #self.var_mean_reward.update_from_moments(mean, std ** 2, count)
        intrinsic_reward = 1 + (intrinsic_reward-self.var_mean_reward.mean) / np.sqrt(self.var_mean_reward.var)# /np.sqrt(self.var_mean_reward.var)
        return intrinsic_reward

    def train(self, data_set):
        s_batch = torch.cat(data_set).to(self.device)
        sample_range = np.arange(len(s_batch))
        forward_mse = torch.nn.MSELoss(reduction='none')
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                predict_next_state_feature, target_next_state_feature = self.forward(s_batch[sample_idx])
                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                self.optimizer.zero_grad()
                loss = forward_loss
                loss.backward()
                self.global_grad_norm_(list(self.predictor.parameters()))
                self.optimizer.step()

    def global_grad_norm_(self, parameters, norm_type=2):
        r"""Clips gradient norm of an iterable of parameters.
        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)

        return total_norm

    class RunningMeanStd(object):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        def __init__(self, epsilon=1e-4, shape=()):
            self.mean = np.zeros(shape, 'float64')
            self.var = np.ones(shape, 'float64')
            self.count = epsilon

        def update(self, x):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

        def update_from_moments(self, batch_mean, batch_var, batch_count):
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count

