import torch
import numpy as np

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
    def __init__(self,device, epsilon=1e-4, num_workers= 16):
        super(RNDModelNet, self).__init__()
        self.device = device
        feature_output = 1024
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, (2, 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(feature_output, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
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
            torch.nn.Linear(feature_output, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
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

    def forward(self, next_obs):
        next_obs = torch.tensor(next_obs).to(self.device)
        next_obs = self.compress_frames(next_obs)
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
    
    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.tensor(next_obs).to(self.device)
        next_obs = self.compress_frames(next_obs).to(self.device)
        target_next_feature = self.target(next_obs).detach()
        predict_next_feature = self.predictor(next_obs).detach()
        intrinsic_reward = ((predict_next_feature - target_next_feature).pow(2).sum(1) / 2).detach().cpu().numpy()
        #intrinsic_reward = torch.dist(predict_next_feature, target_next_feature, 2)
        mean, std, count = np.mean(intrinsic_reward), np.std(intrinsic_reward), len(intrinsic_reward)
        self.var_mean_reward.update_from_moments(mean, std ** 2, count)
        intrinsic_reward = intrinsic_reward /np.sqrt(self.var_mean_reward.var)
        return intrinsic_reward