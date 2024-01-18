import numpy as np
import torch
from torch.optim import Adam
import time
import agent.Effi_UniLight_core as core
import random
# import core2 as core
import copy
import csv

GAMMA = 0.8
# A_LR = 0.0005
# C_LR = 0.0005

A_LR = 0.001
C_LR = 0.001
LAMBDA=0.5
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
EPSILON=0.2
KL_TARGET=0.01



class Effi_UniLight(object):
    ##维度为2 [N,numinter]

    def __init__(self, s_dim=32,old_dim=32, num_intersection=1,num_lanes = 3,num_phases = 8):

        self.obs_dim = s_dim
        self.num_intersection=num_intersection
        self.num_lanes = num_lanes
        self.num_phases = num_phases
        self.old_dim = old_dim

        self.buffer_a = []
        self.buffer_s0 = []
        self.buffer_s1 = []
        self.buffer_s2 = []
        self.buffer_s3 = []
        self.buffer_r = []

        self.ac = core.MLPActorCritic(int(self.obs_dim/self.num_intersection),self.old_dim, self.num_lanes,self.num_phases)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=A_LR)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=C_LR)

    def th_gather_hd(self,x,coords):

        x=x.contiguous()
        inds=coords.mv(torch.LongTensor(x.stride()))
        x_gather=torch.index_select(x.contiguous().view(-1),0,inds)
        return x_gather

    def compute_loss_pi(self, s0, s1, s2,s3, a, adv):


        pi = self.ac.pi(s0, s1, s2,s3, self.num_intersection)
        old_pi=self.ac.oldpi(s0, s1, s2, s3,self.num_intersection)
        pi_resize = torch.reshape(pi, [-1, self.num_phases])
        oldpi_resize = torch.reshape(old_pi, [-1, self.num_phases])


        a_indices = torch.stack(
            [torch.arange(0,(torch.reshape(a, [-1])).shape[0]), torch.reshape(a, [-1])],dim=1)
        a_indices=a_indices.long()


        pi_prob = self.th_gather_hd(pi_resize,a_indices)
        oldpi_prob = self.th_gather_hd(oldpi_resize,a_indices)

        pi_log=torch.log(pi_prob)
        oldpi_log=torch.log(oldpi_prob)

        ratio =torch.reshape((torch.exp(pi_log-oldpi_log)),[-1,1]).mean(dim=1)
        ratio=torch.reshape(ratio,[-1,self.num_intersection])
        # print(ratio)
        adv=torch.reshape(adv,[-1,self.num_intersection])
        clip_adv = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * adv
        loss_pi = -torch.mean(torch.min(ratio * adv, clip_adv))
        return loss_pi


    def choose_action(self,state,inter_need_choose_phase):

        state_input = []
        all_phase_state = []
        mask = []
        old_state = []

        num_inter_need_choose_phase = 0
        for i in range(len(inter_need_choose_phase)):
            if inter_need_choose_phase[i] == True:
                num_inter_need_choose_phase = num_inter_need_choose_phase + 1
                state_input.append(state[0][i])
                all_phase_state.append(state[1][i])
                mask.append(state[2][i])
                old_state.append(state[3][i])
        prob_temp = self.ac.step(torch.as_tensor(state_input, dtype=torch.float32),
                                 torch.as_tensor(all_phase_state, dtype=torch.float32),
                                 torch.as_tensor(mask, dtype=torch.float32),
                                 torch.as_tensor(old_state, dtype=torch.float32),
                                 num_inter_need_choose_phase)
        _action=[]
        need_choose_idx = 0
        for i in range(self.num_intersection):
            if inter_need_choose_phase[i] == False:
                _action.append(0)
            else:
                p = np.array(prob_temp[need_choose_idx])
                action_temp = np.random.choice(range(prob_temp[need_choose_idx].shape[0]),
                                    p=p.ravel())
                _action.append(action_temp)
                need_choose_idx = need_choose_idx + 1

        return _action

    def experience_storage(self, s0, s1,s2, s3,a, r):
        self.buffer_s0.append(s0)
        self.buffer_s1.append(s1)
        self.buffer_s2.append(s2)
        self.buffer_s3.append(s3)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s0, self.buffer_s1,self.buffer_s2,self.buffer_s3,self.buffer_r, self.buffer_a,= [], [], [],[],[],[]

    def trajction_process(self, s_,old_state):
        _s = np.array(s_).reshape([-1,int(self.obs_dim /self.num_intersection)]).tolist()
        v_s_ = self.ac.get_v(torch.as_tensor(_s,dtype=torch.float32),torch.as_tensor(old_state,dtype=torch.float32),self.num_intersection)

        v_s_=v_s_.detach().numpy()
        buffer_r = np.mean(np.array(self.buffer_r).reshape([-1,1]), axis= 1).reshape([-1,int(self.num_intersection)])
        buffer = [[] for i in range(self.num_intersection)]
        for r in buffer_r[::-1]:
            for i in range(int(self.num_intersection)):
                v_s_[i] = (r[i] + GAMMA * v_s_[i])
                buffer[i].append(copy.deepcopy(v_s_[i]))
        for i in range(int(self.num_intersection)):
            buffer[i].reverse()
        out = np.stack([buffer[k] for k in range(int(self.num_intersection))], axis=1)
        self.buffer_r = np.array(out).reshape([-1])

    def update(self,state,old_state):
        self.trajction_process(state,old_state)
        c_s = torch.as_tensor(np.vstack(self.buffer_s0), dtype=torch.float32)
        s1 = torch.as_tensor(np.vstack(self.buffer_s1), dtype=torch.float32)
        s2 = torch.as_tensor(np.vstack(self.buffer_s2), dtype=torch.float32)
        s3 = torch.as_tensor(np.vstack(self.buffer_s3), dtype=torch.float32)

        r = torch.as_tensor(np.vstack(self.buffer_r),dtype=torch.float32)
        a = torch.as_tensor(np.array(self.buffer_a).reshape([-1]),dtype=torch.float32)
        self.ac.update_oldpi()
        adv=self.ac.compute_advantage(c_s,s3,r,self.num_intersection).detach().numpy()
        adv=torch.as_tensor(adv,dtype=torch.float32)

        for i in range(A_UPDATE_STEPS):
            self.pi_optimizer.zero_grad()
            loss_pi= self.compute_loss_pi(c_s,s1,s2,s3,a,adv)
            loss_pi.backward()
            print(loss_pi)
            self.pi_optimizer.step()

        for i in range(C_UPDATE_STEPS):
            self.vf_optimizer.zero_grad()
            adv = self.ac.compute_advantage(c_s,s3, r,self.num_intersection)
            loss_v = torch.square(adv).mean()
            loss_v.backward()
            print(loss_v)
            self.vf_optimizer.step()
        self.empty_buffer()

    def save(self,path):
        torch.save(self.ac, path)
    def load(self,path):
        self.ac = torch.load(path)
