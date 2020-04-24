import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

class PPO_c(nn.Module):
	def __init__(self,state_dim,action_dim,action_std,cuda_on,lr,gamma,K_epoch,lmda,eps_clip):
		super(PPO_c, self).__init__()

		if cuda_on:
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		else:
			device = "cpu"
		self.device = device

		# init
		self.data = []

		# hyperparameters
		self.lr = lr
		self.gamma = gamma
		self.K_epoch = K_epoch
		self.lmbda = lmda
		self.eps_clip = eps_clip

		# actor critic network 
		self.fc1 = nn.Linear(state_dim,32)
		self.fc2 = nn.Linear(32,32)
		self.fc_pi = nn.Linear(32,action_dim)
		self.fc_v = nn.Linear(32,1)

		self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
		self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

	def actor(self,state):
		x = F.tanh(self.fc1(state))
		x = F.tanh(self.fc2(x))
		x = self.fc_pi(x)
		return x

	def critic(self,state):
		x = F.tanh(self.fc1(state))
		x = F.tanh(self.fc2(x))
		x = self.fc_v(x)
		return x

	def policy(self,state):
		
		if type(state) is np.ndarray:
			state = torch.from_numpy(state).float()

		# evaluate the actor network
		action_mean = self.actor(state)
		cov_mat = torch.diag(self.action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		return action, action_logprob

	def evaluate(self, state, action):   
		# get distribution
		action_mean = torch.squeeze(self.actor(state))
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)
		
		action_logprobs = dist.log_prob(torch.squeeze(action))
		dist_entropy = dist.entropy()
		state_value = self.critic(state)
		
		return action_logprobs, torch.squeeze(state_value), dist_entropy

	def put_data(self, transition):
		self.data.append(transition)
		
	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, log_prob_a_lst, done_lst = [], [], [], [], [], []
		for transition in self.data:
			s, a, r, s_prime, log_prob_a, done = transition
			
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			log_prob_a_lst.append([log_prob_a])
			done_mask = 0 if done else 1
			done_lst.append([done_mask])
			
		s = torch.tensor(s_lst, dtype=torch.float)
		a = torch.tensor(a_lst)
		r = torch.tensor(r_lst)
		s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
		done_mask = torch.tensor(done_lst, dtype=torch.float)
		log_prob_a = torch.tensor(log_prob_a_lst)							  
		self.data = []
		return s, a, r, s_prime, done_mask, log_prob_a
		
	def train_net(self):
		s, a, r, s_prime, done_mask, log_prob_a = self.make_batch()

		for i in range(self.K_epoch):
			td_target = r+self.gamma*self.critic(s_prime)*done_mask
			delta = td_target - self.critic(s)
			delta = delta.detach().numpy()

			# make advantage
			advantage_lst = []
			advantage = 0.0
			for delta_t in delta[::-1]:
				advantage = delta_t[0] + \
					self.gamma*self.lmbda*advantage
				advantage_lst.append([advantage])
			advantage_lst.reverse()
			advantage = torch.tensor(advantage_lst, dtype=torch.float)

			# new 
			log_prob_a_new, value, entropy = self.evaluate(s,a)

			ratio = torch.exp(log_prob_a_new - log_prob_a)

			# ppo loss
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1-self.eps_clip, \
				1+self.eps_clip) * advantage
			loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(s) , td_target.detach())

			# update
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()