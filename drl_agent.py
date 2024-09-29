#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:12:28 2023

@author: oscar
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils_ import soft_update, hard_update
from Network import GaussianPolicy, QNetwork, DeterministicPolicy
from cpprb import PrioritizedReplayBuffer

class DRL(object):
    def __init__(self, seed, action_dim, state_dim, pstate_dim, policy_type, critic_type, 
                 LR_A = 1e-3, LR_C = 1e-3, LR_ALPHA=1e-4, BUFFER_SIZE=int(2e5), 
                 TAU=5e-3, GAMMA = 0.99, ALPHA=0.05, POLICY_GUIDANCE=False,
                 VALUE_GUIDANCE = False, ADAPTIVE_CONFIDENCE = True,
                 automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.pstate_dim = pstate_dim
        self.policy_type = policy_type
        self.critic_type = critic_type
        self.gamma = GAMMA
        self.lr_a = LR_A
        self.lr_c = LR_C
        self.lr_alpha = LR_ALPHA
        self.tau = TAU
        self.alpha = ALPHA
        self.itera = 0
        self.demonstration_weight = 1.0
        self.engage_weight = 1.0 #IARL 2.0, others 1.0
        self.buffer_size_expert = 5e3
        self.p_guidance = POLICY_GUIDANCE
        self.v_guidance = VALUE_GUIDANCE
        self.p_loss = 0.0
        self.engage_loss = 0.0
        self.adaptive_weight = ADAPTIVE_CONFIDENCE
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.seed = int(seed)

        self.pre_buffer = False
        self.batch_expert = 0

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE,
                                          {"obs": {"shape": (100,100,9),"dtype": np.uint8},
                                           "pobs": {"shape":pstate_dim},
                                           "act": {"shape":action_dim},
                                           "act_h" : {"shape":action_dim},
                                           "rew": {},
                                           "next_obs": {"shape": (100,100,9),"dtype": np.uint8},
                                           "next_pobs": {"shape":pstate_dim},
                                           "engage": {},
                                           "authority": {},
                                           "done": {}},
                                          next_of=("obs"))

        ######## Initialize Critic Network #########
        self.critic = QNetwork(self.action_dim, self.state_dim, self.pstate_dim).to(device=self.device)
        self.critic_target = QNetwork(self.action_dim, self.state_dim, self.pstate_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_c)
        
        ######### target network update ########
        hard_update(self.critic_target, self.critic)

        ######### Initialize Actor Network #######
        self.policy = GaussianPolicy(self.action_dim, self.state_dim, self.pstate_dim).to(self.device)
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - self.action_dim
            # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.log_alpha = torch.tensor(np.array(np.log(self.alpha )), requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr_alpha)
        
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr_a)

    def choose_action(self, istate, pstate, evaluate=False):
        if istate.ndim < 4:
            istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        
        if evaluate is False:
            action, _, _ = self.policy.sample([istate, pstate])
        else:
            _, _, action = self.policy.sample([istate, pstate])

        return action.detach().squeeze(0).cpu().numpy()

    def learn_guidence(self, batch_size=64):
        agent_buffer_size = self.replay_buffer.get_stored_size()

        ###### For prior demonstration ######
        if self.pre_buffer:
            exp_buffer_size = self.replay_buffer_expert.get_stored_size()
            scale_factor = 1
            # total_size = agent_buffer_size + exp_buffer_size
            
            self.batch_expert = min(np.floor(exp_buffer_size/agent_buffer_size * batch_size / scale_factor), batch_size)

            batch_agent = batch_size
        
        if self.batch_expert > 0:
            demonstration_flag = True
            data_agent = self.replay_buffer.sample(batch_agent)
            data_expert = self.replay_buffer_expert.sample(self.batch_expert)

            istates_agent, pstates_agent, actions_agent, engages = \
                data_agent['obs'], data_agent['pobs'], data_agent['act'], data_agent['engage']
            rewards_agent, next_istates_agent, next_pstates_agent, dones_agent = \
                data_agent['rew'], data_agent['next_obs'], data_agent['next_pobs'], data_agent['done']

            istates_expert, pstates_expert, actions_expert = \
                data_expert['obs'], data_expert['pobs'], data_expert['act_exp']
            rewards_expert, next_istates_expert, next_pstates_expert, dones_expert = \
                data_expert['rew'], data_expert['next_obs'], data_expert['next_pobs'], data_expert['done']

            istates = np.concatenate((istates_agent, istates_expert), axis=0)
            pstates = np.concatenate([pstates_agent, pstates_expert], axis=0)
            actions = np.concatenate([actions_agent, actions_expert], axis=0)
            rewards = np.concatenate([rewards_agent, rewards_expert], axis=0)
            next_istates = np.concatenate([next_istates_agent, next_istates_expert], axis=0)
            next_pstates = np.concatenate([next_pstates_agent, next_pstates_expert], axis=0)
            dones = np.concatenate([dones_agent, dones_expert], axis=0)

        ###### For non prior demonstration #####
        else:
            demonstration_flag = False
            data = self.replay_buffer.sample(batch_size)
            istates, pstates, actions = data['obs'], data['pobs'], data['act']
            rewards, next_istates = data['rew'], data['next_obs']
            next_pstates, dones = data['next_pobs'], data['done']
            human_actions, engages, authorities = data['act_h'], data['engage'], data['authority']
        
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        engages = torch.FloatTensor(engages).to(self.device)
        human_actions = torch.FloatTensor(human_actions).to(self.device)
        authorities =  torch.FloatTensor(authorities).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        ####### Critic Loss Function ######
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ##### Pre buffer demonstration loss #####
        if demonstration_flag:
            istates_expert = torch.FloatTensor(istates_expert).permute(0,3,1,2).to(self.device)
            pstates_expert = torch.FloatTensor(pstates_expert).to(self.device)
            actions_expert = torch.FloatTensor(actions_expert).to(self.device)
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            demonstration_loss = self.demonstration_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            demonstration_loss = 0.0

        ####### Policy Loss Function #######
        pi, log_pi, _ = self.policy.sample([istates, pstates])
        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi).mean()
        self.p_loss = min_qf_pi.detach().cpu().numpy()
        
        ##### Human-in-the-loop loss ######
        if self.p_guidance:
            engage_index = (engages == 1).nonzero(as_tuple=True)[0]
            if engage_index.numel() > 0:
                istates_shared = istates[engage_index]
                pstates_shared = pstates[engage_index]
                target_actions = human_actions[engage_index]
                authority_shared = authorities[engage_index]
                _, _, predicted_actions = self.policy.sample([istates_shared, pstates_shared]) 
                engage_loss = F.mse_loss(predicted_actions, target_actions)
                
                if self.adaptive_weight:
                    engage_loss = (authority_shared * engage_loss).mean() * self.engage_weight
                else:
                    engage_loss = (engage_loss).mean() * self.engage_weight
                
                self.engage_loss = engage_loss.detach().cpu().numpy()
                
                policy_loss = ((self.alpha * log_pi).mean() - min_qf_pi) +\
                                  demonstration_loss + engage_loss
            else:
                authority_shared = 0.0
                engage_loss = 0.0
                policy_loss = (self.alpha * log_pi).mean() - min_qf_pi

        else:
            policy_loss = (self.alpha * log_pi).mean() - min_qf_pi

        print('P loss: ', -self.p_loss, 'Entropy Loss: ',
              (self.alpha * log_pi).mean().detach().cpu().numpy(),
              ', Engage loss: ', self.engage_loss)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Entropy Loss Function #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        ###### Update target network #####
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1
        return qf1_loss.item(), policy_loss.item()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates = data['rew'], data['next_obs']
        next_pstates, dones = data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
                
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        ###### Update target network ######        
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        return qf1_loss.item(), policy_loss.item()
    
    # Define the storing by priority experience replay
    def store_transition(self, s, ps, a, a_h, r, s_, ps_, e, at, d):
        self.replay_buffer.add(obs=s,
                               pobs=ps,
                               act=a,
                               act_h=a_h,
                               rew=r,
                               next_obs=s_,
                               next_pobs=ps_,
                               engage=e,
                               authority=at,
                               done=d)

    # Save and load model parameters
    def load_model(self, output):
        if output is None: return
        self.policy.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save(self, filename, directory, reward, seed):
        torch.save(self.policy.state_dict(), '%s/%s_reward%s_seed%s_actor.pth' % (directory, filename, reward, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward%s_seed%s_critic.pth' % (directory, filename, reward, seed))

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))    

    def load_target(self):
        hard_update(self.critic_target, self.critic)

    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
