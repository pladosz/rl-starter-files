# https://github.com/steffenvan/attentive-multi-tasking
# https://github.com/ysr-plus-ultra/keras_popart_impala

import math

import torch


class PopArtLayer(torch.nn.Module):

    def __init__(self, input_features, output_features, beta=4e-4):
        self.beta = beta

        super(PopArtLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(output_features))

        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):

        normalized_output = inputs.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        return [output, normalized_output]

    def update_parameters(self, vs):

        self.oldmu = self.mu
        self.oldsigma = self.sigma

        #vs = vs * task
        n = vs.shape[0]
        mu = vs.sum() / n
        nu = torch.sum(vs**2) / n
        sigma = torch.sqrt(nu - mu**2)
        sigma = torch.clamp(sigma, min=1e-4, max=1e+6)

        #TODO fix nan check
        if torch.isnan(mu):
            mu = self.mu
        if torch.isnan(sigma):
            sigma = self.sigma
        #mu[torch.isnan(mu)] = self.mu[torch.isnan(mu)]
        #sigma[torch.isnan(sigma)] = self.sigma[torch.isnan(sigma)]

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

    def normalize_weights(self):
        self.weight.data = (self.weight.t() * self.oldsigma / self.sigma).t()
        self.bias.data = (self.oldsigma * self.bias + self.oldmu - self.mu) / self.sigma