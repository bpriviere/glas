
# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

# my package
from learning.deepset import DeepSet

class Consensus_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Consensus_Net, self).__init__()

		if learning_module is "DeepSet":
			self.model = DeepSet(
				param.il_phi_network_architecture,
				param.il_rho_network_architecture,
				param.il_network_activation, 
				param.env_name)

			self.action_dim = 1 
			self.state_dim = 1

		self.a_max = param.a_max
		self.a_min = param.a_min 
		self.agent_memory = param.agent_memory
		self.n_neighbors = param.n_neighbors
		self.n_agents = param.n_agents
		self.bw_threshold = param.il_bw_threshold

	def policy(self,x):

		# inputs observation from all agents
		# outputs policy for all agents
		A = np.empty((len(x),self.action_dim))
		for i,x_i in enumerate(x):
			a_i = self([x_i])
			A[i,:] = a_i.detach().numpy()
		return A

	def __call__(self,x):
		
		# this call takes in a single agents observation and outputs a single agent action
		# observation is the history of relative neighbor measurements 

		a = torch.zeros((len(x),self.action_dim))
		my_relu = nn.ReLU()
		for step,x_step in enumerate(x):

			# belief_weights = self.model(x_step)

			belief_weights = my_relu(self.model(x_step))

			# minimum threshold
			# eps = self.bw_threshold
			eps = 0.02
			belief_weights[torch.where(belief_weights < eps)] = 0

			curr_measurements = torch.zeros((self.n_neighbors))
			for i in range(self.n_neighbors):
				curr_measurements[i] = x_step[i+1][0]

			# print(curr_measurements)
			# print(belief_weights)
			# print(x_step)
			# print(torch.sum( torch.mul(curr_measurements, belief_weights)))
			# exit()

			a[step] = torch.sum( torch.mul(curr_measurements, belief_weights))

			# print('x_step: ', x_step)
			# print('curr_measurements: ', curr_measurements)
			# print('bw: ', belief_weights)
			# # print('bw_no_relu: ', bw_no_relu)
			# print('a[step]: ', a[step])

		return a

	def get_belief_topology(self,x):

		my_relu = nn.ReLU()
		bt = [] 
		for i,x_i in enumerate(x):

			# belief_weights = self.model(x_step)

			belief_weights = my_relu(self.model(x_i))

			# minimum threshold
			# eps = self.bw_threshold
			eps = 0.02
			belief_weights[torch.where(belief_weights < eps)] = 0

			bt.append(belief_weights)
		return bt 