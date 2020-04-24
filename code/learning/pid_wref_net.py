import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

class PID_wRef_Net(nn.Module):
	"""
	neural net to predict gains, kp, kd, from state, s
	"""
	def __init__(self,action_dim,layers,activation):
		super(PID_wRef_Net, self).__init__()
		#self.fc1 = nn.Linear(input_layer, 64)
		#self.fc2 = nn.Linear(64, 64)
		#self.fc3 = nn.Linear(64, input_layer + 4)

		self.layers = layers
		self.activation = activation
		self.state_dim = layers[0].in_features
		self.action_dim = action_dim


	def evalNN(self, x):
		x = torch.from_numpy(np.array(x,ndmin = 2)).float()
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		pid_slice = x[:,0:4]
		ref_slice = x[:,4:]
		x = torch.cat((F.softplus(pid_slice), ref_slice), dim=1)
		return x


	# def evalNN(self, x):
	#	x = torch.from_numpy(array(x,ndmin = 2)).float()
	#	x = F.tanh(self.fc1(x))
	#	x = F.tanh(self.fc2(x))
		# # apply softplus only to PID gains part
	#	x = self.fc3(x)
	#	pid_slice = x[:,0:4]
	#	ref_slice = x[:,4:]
	#	x = torch.cat((F.softplus(pid_slice), ref_slice), dim=1)
	#	return x


	def forward(self, x):
		state = torch.from_numpy(array(x,ndmin = 2)).float()
		x = self.evalNN(x)
		ref_state = x[:,4:]
		error = state-ref_state
		x = x[:,0]*error[:,0] + x[:,1]*error[:,1] + \
			x[:,2]*error[:,2] + x[:,3]*error[:,3] 
		x = x.reshape((len(x),1))
		return x

	def policy(self,state):
		action = self(torch.from_numpy(state).float())
		action = squeeze(action.detach().numpy())
		return action

	def get_kp(self,x):
		x = self.evalNN(x)
		x = x[:,0:2].detach().numpy()
		return x

	def get_kd(self,x):
		x = self.evalNN(x)
		x = x[:,2:4].detach().numpy()
		return x

	def get_ref_state(self,x):
		x = self.evalNN(x)
		x = x[:,4:].detach().numpy()
		return x