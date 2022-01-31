import torch

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class RewGenNet(torch.nn.Module):
    def __init__(self, state_representation_size, device):
        super(RewGenNet, self).__init__()
        #initialize hidden state
        self.device = device
        self.init_hidden()
        self.fc_layer_1 = torch.nn.Linear(state_representation_size, 256)
        self.fc_layer_2 = torch.nn.Linear(256, 128)
        self.fc_layer_3 = torch.nn.Linear(128, 64)
        #self.memory_layer = torch.nn.RNN(64, 32)#weight_init(nn.RNN(64, 32)) #layer_init(nn.Linear(64, 32))
        self.memory_layer = torch.nn.Linear(64, 32)
        self.fc_layer_4 = torch.nn.Linear(32, 1)
        self.to(self.device)
        #self.zero_init()
        self.apply(init_params)


    def forward(self, x):
        #activation = torch.nn.Hardshrink()
        x = torch.relu(self.fc_layer_1(torch.tensor(x).to(self.device)))
        x = torch.relu(self.fc_layer_2(torch.tensor(x).to(self.device)))
        x = torch.relu(self.fc_layer_3(torch.tensor(x).to(self.device)))
        #if len(x.shape) == 1:
        #    x = x[None,None,:]
        #if len(x.shape) == 2:
        #    x = x[None,:]
        #x, self.hidden_state = self.memory_layer(x, self.hidden_state)
        x = self.memory_layer(x)
        self.hidden_state = self.hidden_state.detach()
        x = x.squeeze(0)
        y = torch.tanh(self.fc_layer_4(x))
        return y

    def init_hidden(self, training = False):
        if training == False:
            self.hidden_state = torch.tensor(torch.zeros((1,8,32))).to(self.device)
        else:
            self.hidden_state = torch.tensor(torch.zeros((1,8,32))).to(self.device)
    def reset_hidden(self, agent_id, training = False):
        if training == False:
            self.hidden_state[:,agent_id,:] = torch.tensor(torch.zeros((1,1,32))).to(self.device)
        else:
            self.hidden_state[:,agent_id,:] = torch.tensor(torch.zeros((1,1,32))).to(self.device)
    
    def update_weights(self, reward_net_parameters):
        #update weights per layer:
        self.sum_weight_layer(self.fc_layer_1, reward_net_parameters[0])
        self.sum_weight_layer(self.fc_layer_2, reward_net_parameters[1])
        self.sum_weight_layer(self.fc_layer_3, reward_net_parameters[2])
        self.sum_weight_layer(self.memory_layer, reward_net_parameters[3])  
        self.sum_weight_layer(self.fc_layer_4, reward_net_parameters[4])
        
    def zero_init(self):
        #update weights per layer:
        self.fill_zeros(self.fc_layer_1)
        self.fill_zeros(self.memory_layer)  
        self.fill_zeros(self.fc_layer_2)  
        self.fill_zeros(self.fc_layer_3)
        self.fill_zeros(self.fc_layer_4)  


    def sum_weight_layer(self, layer, new_weights):
        layer_dict = layer.state_dict()
        new_layer_dict = {}
        for param_type in layer_dict:
                numel = torch.numel(layer_dict[param_type])
                param_type_weight_update = new_weights[0:numel]
                param_type_weight_update = torch.reshape(param_type_weight_update,layer_dict[param_type].shape)
                new_layer_dict[param_type] = param_type_weight_update + layer_dict[param_type]
                #delete already used weights
                new_weights = new_weights[numel:]
        if new_weights.shape[0] != 0:
            print("I am error")
            print(new_weights.shape)
            exit()
        layer.load_state_dict(new_layer_dict)

    def fill_zeros(self, layer):
        layer_dict = layer.state_dict()
        new_layer_dict = {}
        for param_type in layer_dict:
                param_type_weight_update = torch.zeros(layer_dict[param_type].shape)
                new_layer_dict[param_type] = param_type_weight_update
        layer.load_state_dict(new_layer_dict)

    def parameter_number(self):
        #linear layers parameters number
        reward_network_params_first_layer = self.parameter_numel_per_layer(self.fc_layer_1)
        reward_network_params_second_layer = self.parameter_numel_per_layer(self.fc_layer_2)
        reward_network_params_third_layer = self.parameter_numel_per_layer(self.fc_layer_3)
        reward_network_memory_layer = self.parameter_numel_per_layer(self.memory_layer)
        reward_network_params_fourth_layer = self.parameter_numel_per_layer(self.fc_layer_4)
        return reward_network_params_first_layer, reward_network_params_second_layer,  reward_network_params_third_layer, reward_network_memory_layer, reward_network_params_fourth_layer
    

    def parameter_numel_per_layer (self,layer):
        layer_dict = layer.state_dict()
        numel_per_layer = []
        for param_type in layer_dict:
            elements = torch.numel(layer_dict[param_type])
            numel_per_layer.append(elements)
        numel_per_layer = np.sum(np.array(numel_per_layer))
        return numel_per_layer
    
    def get_vectorized_param(self):
        parameter_dict = self.state_dict()
        param_list = []
        for parameters_key in parameter_dict:
            param_list.append(torch.flatten(parameter_dict[parameters_key]))
        agent_param = torch.cat(param_list,dim = 0)
        return agent_param