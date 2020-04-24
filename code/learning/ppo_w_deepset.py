
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

# 
from learning.deepset import DeepSet 
# adapted from 
# https://github.com/seungeunrho/minimalRL/blob/master/ppo.py

class PPO_w_DeepSet(nn.Module):
	def __init__(self,
		action_list,
		pi_phi_layers,
		pi_rho_layers, 
		v_phi_layers, 
		v_rho_layers,
		activation,
		cuda_on,
		lr,
		gamma,
		K_epoch,
		lmbda,
		eps_clip):
	
		super(PPO_w_DeepSet, self).__init__()

		self.data = []
		self.actions = action_list 
		self.state_dim = pi_phi_layers[0].in_features
		self.action_dim = pi_rho_layers[-1].out_features

		# hyperparameters
		self.lr = lr
		self.gamma = gamma
		self.K_epoch = K_epoch
		self.lmbda = lmbda
		self.eps_clip = eps_clip

		# network structure
		self.activation = activation
		self.pi_ds = DeepSet(pi_phi_layers,pi_rho_layers,activation)
		self.v_ds = DeepSet(v_phi_layers,v_rho_layers,activation)

		# 
		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def pi(self, x, softmax_dim = 0):
		prob = F.softmax(self.pi_ds(x),dim=softmax_dim) 
		return prob
	
	def v(self, x):
		return self.v_ds(x)
	  
	def put_data(self, transition):
		self.data.append(transition)
		
	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
		for transition in self.data:
			s, a, r, s_prime, prob_a, done = transition
			
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			prob_a_lst.append([prob_a])
			done_mask = 0 if done else 1
			done_lst.append([done_mask])
			
		s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
										  torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
										  torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
		self.data = []
		return s, a, r, s_prime, done_mask, prob_a
		
	def train_net(self):

		s, a, r, s_prime, done_mask, prob_a = self.make_batch()

		for i in range(self.K_epoch):

			td_target = r + self.gamma * self.v(s_prime) * done_mask
			delta = td_target - self.v(s)
			delta = delta.detach().numpy()

			advantage_lst = []
			advantage = 0.0
			for delta_t in delta[::-1]:
				advantage = self.gamma * self.lmbda * advantage + delta_t[0]
				advantage_lst.append([advantage])
			advantage_lst.reverse()
			advantage = torch.tensor(advantage_lst, dtype=torch.float)

			pi = self.pi(s, softmax_dim=1)
			pi_a = pi.gather(1,a)
			ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
			loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

	def policy(self, x):
		# inputs observation from all agents...
		# outputs policy for all agents

		self.action_dim_per_agent = 1 

		A = np.empty((len(x),self.action_dim_per_agent))
		for i,x_i in enumerate(x):
			prob = self.pi([x_i])
			m = Categorical(prob)
			classification = m.sample().item()
			A[i,:] = self.class_to_action(classification)
		return A

	def train_policy(self,x):

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.action_dim))
		for i,x_i in enumerate(x):
			a_i = self([x_i])
			A[i,:] = a_i.detach().numpy()
		return A


	def class_to_action(self, a):
		return self.actions[a]