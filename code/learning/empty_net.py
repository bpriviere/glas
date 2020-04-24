
# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import concurrent

# my package
from learning.deepset import DeepSet
from learning.feedforward import FeedForward

class Empty_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Empty_Net, self).__init__()

		if learning_module is "DeepSet":

			self.model_neighbors = DeepSet(
				param.il_phi_network_architecture,
				param.il_rho_network_architecture,
				param.il_network_activation,
				param.env_name
				)
			self.model_obstacles = DeepSet(
				param.il_phi_obs_network_architecture,
				param.il_rho_obs_network_architecture,
				param.il_network_activation,
				param.env_name
				)
			self.psi = FeedForward(
				param.il_psi_network_architecture,
				param.il_network_activation)

		self.param = param
		self.device = torch.device('cpu')

		self.dim_neighbor = param.il_phi_network_architecture[0].in_features
		self.dim_action = param.il_psi_network_architecture[-1].out_features
		self.dim_state = param.il_psi_network_architecture[0].in_features - \
						param.il_rho_network_architecture[-1].out_features - \
						param.il_rho_obs_network_architecture[-1].out_features


	def to(self, device):
		self.device = device
		self.model_neighbors.to(device)
		self.model_obstacles.to(device)
		self.psi.to(device)
		return super().to(device)

	def policy(self,x):

		# inputs observation from all agents...
		# outputs policy for all agents

		A = np.empty((len(x),self.dim_action))
		for i,x_i in enumerate(x):
			a_i = self(torch.Tensor(x_i))
			a_i = a_i.detach().numpy()
			A[i,:] = a_i
		return A

	def export_to_onnx(self, filename):
		self.model_neighbors.export_to_onnx("{}_neighbors".format(filename))
		self.model_obstacles.export_to_onnx("{}_obstacles".format(filename))
		self.psi.export_to_onnx("{}_psi".format(filename))

	def get_num_neighbors(self,x):
		return int(x[0,0])

	def get_num_obstacles(self,x):
		nn = self.get_num_neighbors(x)
		return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

	def get_agent_idx_all(self,x):
		nn = self.get_num_neighbors(x)
		idx = np.arange(1+self.dim_state,1+self.dim_state+self.dim_neighbor*nn,dtype=int)
		return idx

	def get_obstacle_idx_all(self,x):
		nn = self.get_num_neighbors(x)
		idx = np.arange((1+self.dim_state)+self.dim_neighbor*nn, x.size()[1],dtype=int)
		return idx

	def get_goal_idx(self,x):
		idx = np.arange(1,1+self.dim_state,dtype=int)
		return idx 

	def __call__(self,x):
		# batches are grouped by number of neighbors (i.e., each batch has data with the same number of neighbors)
		# x is a 2D tensor, where the columns are: relative_goal, relative_neighbors, ...

		num_neighbors = int(x[0,0]) #int((x.size()[1]-4)/4)
		num_obstacles = int((x.size()[1] - (1 + self.dim_state + self.dim_neighbor*num_neighbors))/2)

		rho_neighbors = self.model_neighbors.forward(x[:,self.get_agent_idx_all(x)])
		rho_obstacles = self.model_obstacles.forward(x[:,self.get_obstacle_idx_all(x)])
		g = x[:,self.get_goal_idx(x)]

		x = torch.cat((rho_neighbors, rho_obstacles, g),1)
		x = self.psi(x)

		return x
