import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

# adapted from 
# https://github.com/seungeunrho/minimalRL/blob/master/ppo.py

class PPO(nn.Module):
	def __init__(self,
		action_list,
		state_dim,
		action_dim,
		action_std,
		cuda_on,
		lr,
		gamma,
		K_epoch,
		lmda,
		eps_clip):
	
		super(PPO, self).__init__()

		self.data = []
		self.actions = action_list 

		# hyperparameters
		self.lr = lr
		self.gamma = gamma
		self.K_epoch = K_epoch
		self.lmbda = lmda
		self.eps_clip = eps_clip

		self.fc1   = nn.Linear(state_dim,32)
		self.fc2   = nn.Linear(32,32)
		self.fc_pi = nn.Linear(32,len(self.actions))
		self.fc_v  = nn.Linear(32,1)
		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def pi(self, x, softmax_dim = 0):
		state = x
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = self.fc_pi(x)
		prob = F.softmax(x, dim=softmax_dim)
		return prob
	
	def v(self, x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		v = self.fc_v(x)
		return v
	  
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

	def policy(self, state):
		prob = self.pi(torch.from_numpy(state).float())
		m = Categorical(prob)
		classification = m.sample().item()
		return self.class_to_action(classification)

	def class_to_action(self, a):
		return self.actions[a]

	def get_optimizers(self):
		return [self.optimizer]