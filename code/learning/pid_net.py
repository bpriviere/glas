import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from numpy import squeeze, array,arange, linspace
import numpy as np 

class PID_Net(nn.Module):
	"""
	neural net to predict gains, kp, kd, from state, s
	"""
	def __init__(self, input_layer):
		super(PID_Net, self).__init__()
		self.fc1 = nn.Linear(input_layer, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 4)

	def evalNN(self, x):
		x = torch.from_numpy(array(x,ndmin = 2)).float()
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x = F.softplus(self.fc3(x))
		return x

	def forward(self, x):
		state = torch.from_numpy(array(x,ndmin = 2)).float()
		x = self.evalNN(x)
		error = state
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