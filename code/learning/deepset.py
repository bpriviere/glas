
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

		if self.env_name == 'Consensus':
			return self.consensus_forward(x)
		elif self.env_name in ['SingleIntegrator','DoubleIntegrator','SingleIntegratorVelSensing']:
			return self.si_forward(x)

	def consensus_forward(self,x):

		# x is a relative neighbor histories 
		# RHO_IN = torch.zeros((1,self.rho.in_dim))

		summ = torch.zeros((self.phi.out_dim))
		for step_rnh, rnh in enumerate(x):

			if step_rnh == 0:
				self_history = np.array(rnh, ndmin=1)
				self_history = torch.from_numpy(self_history).float()
			else:
				rnh = np.array(rnh, ndmin=1)
				rnh = torch.from_numpy(rnh).float()
				summ += self.phi(rnh)

		# print(self_history.shape)
		# print(summ.shape)
		# print(torch.cat((self_history,summ)))
		# exit()

		RHO_IN = torch.cat((self_history,summ))
		RHO_OUT = self.rho(RHO_IN)
		return RHO_OUT

	def si_forward(self,x):
		# print(x)
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

