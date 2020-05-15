
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from learning.feedforward import FeedForward

class DeepSet(nn.Module):

	def __init__(self,phi_layers,rho_layers,activation,env_name):
		super(DeepSet, self).__init__()
		
		self.phi = FeedForward(phi_layers,activation)
		self.rho = FeedForward(rho_layers,activation)
		self.env_name = env_name
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.phi.to(device)
		self.rho.to(device)
		return super().to(device)

	def export_to_onnx(self, filename):
		self.phi.export_to_onnx("{}_phi".format(filename))
		self.rho.export_to_onnx("{}_rho".format(filename))

	def forward(self,x):
		X = torch.zeros((len(x),self.rho.in_dim), device=self.device)
		num_elements = int(x.size()[1] / self.phi.in_dim)
		for i in range(num_elements):
			X += self.phi(x[:,i*self.phi.in_dim:(i+1)*self.phi.in_dim])
		return self.rho(X)


class DeepSetObstacles(nn.Module):

	def __init__(self,phi_layers,rho_layers,activation,env_name):
		super(DeepSetObstacles, self).__init__()
		
		self.phi = FeedForward(phi_layers,activation)
		self.rho = FeedForward(rho_layers,activation)
		self.env_name = env_name
		self.device = torch.device('cpu')

	def to(self, device):
		self.device = device
		self.phi.to(device)
		self.rho.to(device)
		return super().to(device)

	def export_to_onnx(self, filename):
		self.phi.export_to_onnx("{}_phi".format(filename))
		self.rho.export_to_onnx("{}_rho".format(filename))

	def forward(self, x, vel):
		# print(x)
		X = torch.zeros((len(x),self.rho.in_dim), device=self.device)
		if self.phi.in_dim == 4:
			# In this case, we also add our own velocity information
			num_elements = int(x.size()[1] / 2)
			for i in range(num_elements):
				X += self.phi(torch.cat((x[:,i*2:(i+1)*2], vel), dim=1))
			return self.rho(X)
		elif self.phi.in_dim == 2:
			# regular case: only relative positions
			num_elements = int(x.size()[1] / self.phi.in_dim)
			for i in range(num_elements):
				X += self.phi(x[:,i*self.phi.in_dim:(i+1)*self.phi.in_dim])
			return self.rho(X)
		else:
			print('unknown phi.in_dim!', self.phi.in_dim)
			exit()

