
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np 


class Ref_Net(nn.Module):
	"""
	neural net to state_ref
	pi(s) = a
	where last layer:
	a = kp (s - s_ref) + kd (sd - sd_ref)
	"""
	def __init__(self, state_dim, action_dim, a_min, a_max, Kp, Kd, layers, activation):
		super(Ref_Net, self).__init__()
		self.a_min = torch.from_numpy(a_min).float()
		self.a_max = torch.from_numpy(a_max).float()
		self.Kp = Kp
		self.Kd = Kd
		self.layers = layers
		self.activation = activation
		self.n = state_dim
		self.m = action_dim

	def evalNN(self, x):
		x = torch.from_numpy(np.array(x,ndmin = 2)).float()		
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		return self.layers[-1](x)

	def forward(self, x):
		# input: 
		# 	x, nd array, (n,)
		# output:
		# 	a, nd array, (m,1)

		# batch input: 
		# 	x, torch tensor, (ndata,n)
		# 	a, torch tensor, (ndata,m)

		state = torch.from_numpy(np.array(x,ndmin = 2)).float()
		ref_state = self.evalNN(x)

		# error (proportional and derivative)
		error = state-ref_state
		ep = error[:,0:int(self.n/2)]
		ed = error[:,int(self.n/2):]
		
		# gain matrix 
		Kp = torch.tensor(self.Kp*np.ones((self.m,int(self.n/2)))).float()
		Kd = torch.tensor(self.Kd*np.ones((self.m,int(self.n/2)))).float()

		# PD control 
		a = (torch.mm(Kp,ep.T) + torch.mm(Kd,ed.T)).T

		# clamp
		a = torch.tanh(a)
		a = (a+1)/2*(self.a_max-self.a_min)+self.a_min
		return a


	def policy(self,state):

		state = state[0]
		action = self(torch.from_numpy(state).float())
		action = np.squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		return self.Kp

	def get_kd(self,x):
		return self.Kd

	def get_ref_state(self,x):
		x = self.evalNN(x)
		x = x.detach().numpy()
		return x