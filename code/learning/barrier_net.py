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
# from learning.empty_net import Empty_Net
from learning.feedforward import FeedForward
from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle, min_point_circle_rectangle


class Barrier_Net(nn.Module):

	def __init__(self,param,learning_module):
		super(Barrier_Net, self).__init__()
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
		
		# temp fix 
		# self.action_dim_per_agent = param.il_psi_network_architecture[-1].out_features
		# self.state_dim_per_agent = param.il_phi_network_architecture[0].in_features

		self.layers = param.il_psi_network_architecture
		self.activation = param.il_network_activation
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
		return super().to(device)

	def policy(self,x):
		# only called in rollout 

		self.to("cpu")

		A = np.empty((len(x),self.dim_action))
		for i,x_i in enumerate(x):
			a_i = self(x_i)
			A[i,:] = a_i
		return A

	def __call__(self,x):

		if type(x) == torch.Tensor:

			if self.param.safety is "potential":
				P,H = self.torch_get_relative_positions_and_safety_functions(x)
				barrier_action = self.torch_get_barrier_action(x,P,H)
				empty_action = self.empty(x)
				empty_action = self.torch_scale(empty_action, self.param.pi_max)
				adaptive_scaling = self.torch_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = torch.mul(adaptive_scaling,empty_action)+barrier_action 
				action = self.torch_scale(action, self.param.a_max)

			elif self.param.safety is "fdbk":
				P,H = self.torch_get_relative_positions_and_safety_functions(x)
				Psi = self.torch_get_psi(x,P,H)
				GradPsiInv = self.torch_get_grad_psi_inv(x,P,H)
				barrier_action = -1*self.param.b_gamma*torch.mul(Psi.unsqueeze(1),GradPsiInv)
				empty_action = self.empty(x)
				empty_action = self.torch_scale(empty_action, self.param.pi_max)
				alpha_fdbk = self.torch_get_alpha_fdbk()
				action = alpha_fdbk*empty_action + barrier_action 
				action = self.torch_scale(action, self.param.a_max)

		elif type(x) is np.ndarray:

			if self.param.safety is "potential":
				P,H = self.numpy_get_relative_positions_and_safety_functions(x)
				barrier_action = self.numpy_get_barrier_action(x,P,H)
				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.numpy_scale(empty_action, self.param.pi_max)
				adaptive_scaling = self.numpy_get_adaptive_scaling(x,empty_action,barrier_action,P,H)
				action = adaptive_scaling*empty_action+barrier_action 
				action = self.numpy_scale(action, self.param.a_max)

			elif self.param.safety is "fdbk":
				P,H = self.numpy_get_relative_positions_and_safety_functions(x)
				Psi = self.numpy_get_psi(x,P,H)
				GradPsiInv = self.numpy_get_grad_psi_inv(x,P,H)
				barrier_action = -1*self.param.b_gamma*Psi*GradPsiInv
				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.numpy_scale(empty_action, self.param.pi_max)
				alpha_fdbk = self.numpy_get_alpha_fdbk()
				action = alpha_fdbk*empty_action + barrier_action 
				action = self.numpy_scale(action, self.param.a_max)

		else:
			exit('type(x) not recognized: ', type(x))

		return action 

	def empty(self,x):
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


	# torch functions, optimzied for batch 
	def torch_get_psi(self,x,P,H):
		psi = torch.zeros((len(x)),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			psi += -np.log(H[:,j])
		return psi 

	def torch_get_grad_psi_inv(self,x,P,H):
		barrier = self.torch_get_barrier_action(x,P,H)
		barrier += self.param.eps_h*torch.rand(barrier.shape)
		grad_psi_inv = torch.ones(barrier.shape,device=self.device)
		grad_psi_inv[:,0] = torch.mul(grad_psi_inv[:,1]-barrier[:,1],torch.pow(barrier[:,0],-1))
		return grad_psi_inv

	def torch_get_alpha_fdbk(self):
		phi_max = -self.param.n_agents**2.0*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
		alpha = phi_max*self.param.b_gamma 
		return alpha 


	def torch_get_relative_positions_and_safety_functions(self,x):

		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x)

		P = torch.zeros((nd,nn+no,2),device=self.device) # pj - pi 
		H = torch.zeros((nd,nn+no),device=self.device) 
		curr_idx = 0

		for j in range(nn):
			# j+1 to skip relative goal entries, +1 to skip number of neighbors column
			idx = self.get_agent_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx]
			H[:,curr_idx] = torch.max(torch.norm(x[:,idx], p=2, dim=1) - 2*self.param.r_agent, torch.zeros(1,device=self.device))
			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx]
			closest_point = torch_min_point_circle_rectangle(
				torch.zeros(2,device=self.device), 
				self.param.r_agent,
				-x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
				-x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
			H[:,curr_idx] = torch.max(torch.norm(closest_point, p=2, dim=1) - self.param.r_agent, torch.zeros(1,device=self.device))
			curr_idx += 1

		return P,H 

	def torch_get_barrier_action(self,x,P,H):
		barrier = torch.zeros((len(x),self.dim_action),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			barrier += self.torch_get_barrier(P[:,j,:],H[:,j])
		return barrier

	def torch_get_barrier(self,P,H):
		normP = torch.norm(P,p=2,dim=1)
		normP = normP.unsqueeze(1)
		normP = torch_tile(normP,1,P.shape[1])
		H = H.unsqueeze(1)
		H = torch_tile(H,1,P.shape[1])
		barrier = -1*self.param.b_gamma*torch.mul(torch.mul(torch.pow(H,-1),torch.pow(normP,-1)),P)
		return barrier		

	def torch_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
		adaptive_scaling = torch.ones(H.shape[0],device=self.device)
		# print('H',H)
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]
			normb = torch.norm(barrier_action,p=2,dim=1)
			normpi = torch.norm(empty_action,p=2,dim=1)
			adaptive_scaling[minH < self.param.Delta_R] = torch.min(\
				torch.mul(normb,torch.pow(normpi,-1)),torch.ones(1,device=self.device))[0]
		return adaptive_scaling.unsqueeze(1)

	def torch_scale(self,action,max_action):
		inv_alpha = action.norm(p=2,dim=1)/max_action
		inv_alpha = torch.clamp(inv_alpha,min=1)
		inv_alpha = inv_alpha.unsqueeze(0).T
		inv_alpha = torch_tile(inv_alpha,1,2)
		action = action*inv_alpha.pow_(-1)
		return action

	# numpy function, otpimized for rollout
	def numpy_get_psi(self,x,P,H):
		psi = np.zeros(1,dtype=float)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			psi += -np.log(H[:,j])
		return psi 

	def numpy_get_grad_psi_inv(self,x,P,H):
		barrier = self.numpy_get_barrier_action(x,P,H)
		barrier += self.param.eps_h*np.random.random(barrier.shape)
		grad_psi_inv = np.ones(barrier.shape)
		grad_psi_inv[:,0] = (1-barrier[:,1])/barrier[:,0]
		return grad_psi_inv

	def numpy_get_alpha_fdbk(self):
		phi_max = -self.param.n_agents**2.0*np.log(self.param.Delta_R/(self.param.r_obs_sense-self.param.r_agent))
		alpha = phi_max*self.param.b_gamma 
		return alpha 

	def numpy_get_relative_positions_and_safety_functions(self,x):
		
		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x) 

		P = np.zeros((nd,nn+no,2)) # pj - pi 
		H = np.zeros((nd,nn+no)) 
		curr_idx = 0

		for j in range(nn):
			idx = self.get_agent_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx]
			H[:,curr_idx] = np.max((np.linalg.norm(x[:,idx]) - 2*self.param.r_agent, np.zeros(1)))
			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx]
			closest_point = min_point_circle_rectangle(
				np.zeros(2), 
				self.param.r_agent,
				-x[:,idx] - np.array([0.5,0.5]), 
				-x[:,idx] + np.array([0.5,0.5]))
			H[:,curr_idx] = np.max((np.linalg.norm(closest_point) - self.param.r_agent, np.zeros(1)))
			curr_idx += 1
		return P,H 

	def numpy_get_barrier_action(self,x,P,H):
		barrier = np.zeros((len(x),self.dim_action))
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			barrier += self.numpy_get_barrier(P[:,j,:],H[:,j])
		return barrier

	def numpy_get_barrier(self,P,H):
		normp = np.linalg.norm(P)
		barrier = -1*self.param.b_gamma/(H*normp)*P
		return barrier

	def numpy_get_adaptive_scaling(self,x,empty_action,barrier_action,P,H):
		adaptive_scaling = 1.0 
		if not H.size == 0 and np.min(H) < self.param.Delta_R:
			normb = np.linalg.norm(barrier_action)
			normpi = np.linalg.norm(empty_action)
			adaptive_scaling = np.min((normb/normpi,1))
		return adaptive_scaling

	def numpy_scale(self,action,max_action):
		alpha = max_action/np.linalg.norm(action)
		alpha = np.min((alpha,1))
		action = action*alpha
		return action


	# helper fnc		

	def get_num_neighbors(self,x):
		return int(x[0,0])

	def get_num_obstacles(self,x):
		nn = self.get_num_neighbors(x)
		return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

	def get_agent_idx_j(self,x,j):
		idx = 1+self.dim_state + self.dim_neighbor*j+np.arange(0,2,dtype=int)
		return idx

	def get_obstacle_idx_j(self,x,j):
		nn = self.get_num_neighbors(x)
		idx = 1 + self.dim_state + self.dim_neighbor*nn+j*2+np.arange(0,2,dtype=int)
		return idx

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