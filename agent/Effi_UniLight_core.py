import math

import numpy as np

import torch
import torch.nn as nn
import sys
sys.path.append("tcnmaster")

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs,intersection,raw,col, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class Actor_front2(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super(Actor_front2, self).__init__()

        weight_tensor=torch.randn(obs_dim,hidden_sizes)
        bais_tensor=torch.randn(hidden_sizes)
        self.weights=nn.Parameter(weight_tensor,requires_grad=True)
        self.bais=nn.Parameter(bais_tensor,requires_grad=True)
        nn.init.normal_(self.weights, mean=0, std=1)
        nn.init.constant_(self.bais, val=0.1)


    def forward(self, x):
        return torch.matmul(x,self.weights)+self.bais

class MLPCategoricalActor(Actor):
    def __init__(self, state_input_dim,old_dim,num_lanes, num_phases, hidden_sizes):
        super().__init__()
        self.state_input_dim=state_input_dim
        self.num_lanes = num_lanes
        self.num_phases=num_phases
        self.hidden_sizes=hidden_sizes
        self.MLP1=Actor_front2(state_input_dim, hidden_sizes)
        self.LSTM1 = nn.LSTM(old_dim, hidden_sizes, batch_first=True)
        self.ATTENTION = nn.MultiheadAttention(hidden_sizes,4,batch_first=True)
        self.LSTM2 = nn.LSTM(hidden_sizes, hidden_sizes, batch_first=True, bidirectional=False)
        self.MLP2 = Actor_front2(num_lanes, hidden_sizes)
        self.MLP3 = Actor_front2(self.num_lanes, hidden_sizes)
        self.MLP_output = Actor_front2(hidden_sizes, 1)



    def mask_softmax(self, x, mask):
        x = torch.exp(x)
        x = torch.mul(x, torch.tensor(mask))
        sum = torch.sum(x, dim=1)
        sum = torch.unsqueeze(sum, dim=1)
        sum = [sum for i in range(len(mask[0]))]
        sum = torch.cat(sum, dim=1)
        x = torch.div(x, torch.as_tensor(sum))
        return x



    def forward(self, state_input,all_phase_state, mask,old_state,intersection):

        #Spatial-Temporal Feature Learning:
        state_input = torch.reshape(state_input, [-1, self.state_input_dim])
        state_input = self.MLP1(state_input)
        out_old_state,_ = self.LSTM1(old_state)
        state_input = torch.unsqueeze(state_input,dim = 1)

        attention,_ = self.ATTENTION(state_input,out_old_state,out_old_state)

        #High-Dimension Decomposition
        ST=torch.reshape(attention, [-1, intersection, self.hidden_sizes])
        batch = len(ST)
        h0=torch.zeros(1,batch,self.hidden_sizes)
        c0=torch.zeros(1 ,batch,self.hidden_sizes)
        o,_=self.LSTM2(ST, (h0, c0))
        o=torch.reshape(o, [-1, self.hidden_sizes])


        #Phase Decision
        all_phase_state = self.MLP3(all_phase_state)
        all_phase_state_shape = all_phase_state.shape
        o = torch.stack([o for i in range(all_phase_state_shape[1])],axis = 1)
        o = torch.reshape(o,[-1,self.hidden_sizes])
        all_phase_state = torch.reshape(all_phase_state, [-1, self.hidden_sizes])

        c = torch.mul(all_phase_state,o)
        d = self.MLP_output(c)
        d = torch.reshape(d,[-1,self.num_phases])

        output = self.mask_softmax(d, mask)
        return output

    def _log_prob_from_distribution(self, pi, act):
            return pi.log_prob(act)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim,old_dim, num_lanes,hidden_sizes, activation):
        super().__init__()
        self.l1 = nn.Linear(obs_dim,100)
        a=self.l1.weight
        nn.init.normal_(a,mean=0,std=0.01)
        b=self.l1.bias
        nn.init.constant_(b,val=0.01)
        self.v= nn.Linear(100,1)
        self.v_net=nn.Sequential(self.l1, self.v)
        self.LSTM = nn.LSTM(old_dim, 100, batch_first=True)

        self.ATTENTION = nn.MultiheadAttention(100,4,batch_first=True)

    def forward(self, obs,old_state,intersection):
        out1,_ = self.LSTM(old_state)
        obs = self.l1(obs)
        obs = torch.unsqueeze(obs,1)
        attention,_ = self.ATTENTION(obs,out1,out1)
        attention = torch.squeeze(attention)
        return self.v(attention)

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space,old_dim, num_lanes,num_phases,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.obs_dim = observation_space
        self.v = MLPCritic(self.obs_dim,old_dim,num_lanes, hidden_sizes, activation)
        self.num_lanes = num_lanes
        self.num_phases = num_phases

        self.pi = MLPCategoricalActor(self.obs_dim,old_dim,self.num_lanes, self.num_phases, hidden_sizes[0])
        self.oldpi=MLPCategoricalActor(self.obs_dim,old_dim,self.num_lanes, self.num_phases, hidden_sizes[0])
        for k,value in self.oldpi.named_parameters():
            value.requires_grad=False
    def step(self, state_input,all_phase_state,mask,old_state,intersection):
        with torch.no_grad():
            pi = self.pi(state_input,all_phase_state,mask,old_state, intersection)
        return pi

    def get_v(self,obs,old_state,intersection):
        v = self.v(obs,old_state,intersection)
        return v

    def compute_advantage(self,state,old_state,reward,intersection):
        v=self.v(state,old_state,intersection)
        v=torch.reshape(v,[-1,1])
        return reward-v

    def update_oldpi(self):
        print(self.oldpi.load_state_dict)
        self.oldpi.load_state_dict(self.pi.state_dict())