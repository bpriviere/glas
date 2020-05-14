
import torch 
import numpy as np 
import cvxpy as cp
from cvxpy.atoms.norm_inf import norm_inf
from collections import namedtuple


from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle, min_point_circle_rectangle
from barrier_fncs import Barrier_Fncs


# motion planning 

class Empty_Net_wAPF():

	def __init__(self,param,env,empty):

		self.env = env
		self.empty = empty
		self.param = param 
		self.bf = Barrier_Fncs(param)
		self.device = torch.device('cpu')
		self.dim_neighbor = param.il_phi_network_architecture[0].in_features
		self.dim_action = param.il_psi_network_architecture[-1].out_features
		self.dim_state = param.il_psi_network_architecture[0].in_features - \
						param.il_rho_network_architecture[-1].out_features - \
						param.il_rho_obs_network_architecture[-1].out_features


	def __call__(self,x):

		if type(x) == torch.Tensor:

			if self.param.safety == "cf_si_2":

				P,H = self.bf.torch_get_relative_positions_and_safety_functions(x)
				barrier_action = self.bf.torch_fdbk_si(x,P,H)

				empty_action = self.empty(x)
				empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

				cf_alpha = self.bf.torch_get_cf_si_2(x,empty_action,barrier_action,P,H)
				action = torch.mul(cf_alpha,empty_action) + torch.mul(1-cf_alpha,barrier_action)
				action = self.bf.torch_scale(action, self.param.a_max)				

			elif self.param.safety == "cf_di_2":

				P,H = self.bf.torch_get_relative_positions_and_safety_functions(x)
				barrier_action = self.bf.torch_fdbk_di(x,P,H)

				empty_action = self.empty(x)
				empty_action = self.bf.torch_scale(empty_action, self.param.pi_max)

				cf_alpha = self.bf.torch_get_cf_di_2(x,empty_action,barrier_action,P,H)
				action = torch.mul(cf_alpha,empty_action) + torch.mul(1-cf_alpha,barrier_action)
				action = self.bf.torch_scale(action, self.param.a_max)

			else:
				exit('self.param.safety: {} not recognized'.format(self.param.safety))


		elif type(x) is np.ndarray:

			if self.param.safety == "cf_si_2":

				P,H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
				barrier_action = self.bf.numpy_fdbk_si(x,P,H)

				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

				cf_alpha = self.bf.numpy_get_cf_si_2(x,P,H,empty_action,barrier_action)
				action = cf_alpha*empty_action + (1-cf_alpha)*barrier_action 
				action = self.bf.numpy_scale(action, self.param.a_max)				

			elif self.param.safety == "cf_di_2":

				P,H = self.bf.numpy_get_relative_positions_and_safety_functions(x)
				barrier_action = self.bf.numpy_fdbk_di(x,P,H)

				empty_action = self.empty(torch.tensor(x).float()).detach().numpy()
				empty_action = self.bf.numpy_scale(empty_action, self.param.pi_max)

				cf_alpha = self.bf.numpy_get_cf_di_2(x,P,H,empty_action,barrier_action)
				action = cf_alpha*empty_action + (1-cf_alpha)*barrier_action 
				action = self.bf.numpy_scale(action, self.param.a_max)				

			else:
				exit('self.param.safety: {} not recognized'.format(self.param.safety))

		else:
			exit('type(x) not recognized: ', type(x))

		return action 

	def policy(self,x):

		if self.param.rollout_batch_on:
			grouping = dict()
			for i,x_i in enumerate(x):
				key = (int(x_i[0][0]), x_i.shape[1])
				if key in grouping:
					grouping[key].append(i)
				else:
					grouping[key] = [i]

			if len(grouping) < len(x):
				A = np.empty((len(x),self.dim_action))
				for key, idxs in grouping.items():
					batch = torch.Tensor([x[idx][0] for idx in idxs])
					a = self(batch)
					a = a.detach().numpy()
					for i, idx in enumerate(idxs):
						A[idx,:] = a[i]

				return A
			else:
				A = np.empty((len(x),self.dim_action))
				for i,x_i in enumerate(x):
					a_i = self(x_i)
					A[i,:] = a_i 
				return A
		else:
			A = np.empty((len(x),self.dim_action))
			for i,x_i in enumerate(x):
				a_i = self(x_i)
				A[i,:] = a_i 
			return A

# other

class ZeroPolicy:
	def __init__(self,env):
		self.env = env
	def policy(self,state):
		return torch.zeros((self.env.m))
	def __call__(self,x):
		return torch.zeros((len(x),2))

class GoToGoalPolicy:
	def __init__(self,param,env):
		self.param = param
		self.env = env

	def policy(self, o):
		A = np.empty((len(o),self.env.action_dim_per_agent))
		for i,o_i in enumerate(o):
			a_i = self(o_i)
			A[i,:] = a_i 
		return A

	def __call__(self, o):
		A = torch.empty((len(o),self.env.action_dim_per_agent))
		for i, observation_i in enumerate(o):
			relative_goal = np.array(observation_i[1:3])
			relative_vel = np.array(observation_i[3:5])
			
			a_nom = self.param.cbf_kp*relative_goal
			if hasattr(self.env.param,'env_name') and self.env.param.env_name == "DoubleIntegrator":
				a_nom += self.param.cbf_kd*relative_vel
			A[i,:] = torch.tensor(a_nom)
		return A

