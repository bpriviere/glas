
# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

from utilities import torch_tile, min_dist_circle_rectangle, torch_min_point_circle_rectangle, min_point_circle_rectangle


class Barrier_Fncs():
	def __init__(self,param):
		self.param = param 
		self.device = torch.device('cpu')

		self.dim_neighbor = param.il_phi_network_architecture[0].in_features
		self.dim_action = param.il_psi_network_architecture[-1].out_features
		self.dim_state = param.il_psi_network_architecture[0].in_features - \
						param.il_rho_network_architecture[-1].out_features - \
						param.il_rho_obs_network_architecture[-1].out_features

		self.device = torch.device('cpu')

		self.sigmoid = torch.nn.Sigmoid()

	def to(self,device):
		self.device = device

	# torch
	def torch_get_relative_positions_and_safety_functions(self,x):

		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x)

		P = torch.zeros((nd,nn+no,2),device=self.device) # pj - pi 
		H = torch.zeros((nd,nn+no),device=self.device) 
		curr_idx = 0

		for j in range(nn):
			idx = self.get_agent_pos_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent * torch.pow(torch.norm(x[:,idx], p=2, dim=1).unsqueeze(1), -1))
			H[:,curr_idx] = (torch.norm(P[:,curr_idx,:], p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)
			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			closest_point = torch_min_point_circle_rectangle(
				torch.zeros(2,device=self.device), 
				self.param.r_agent,
				x[:,idx] - torch.tensor([0.5,0.5],device=self.device), 
				x[:,idx] + torch.tensor([0.5,0.5],device=self.device))
			P[:,curr_idx,:] = closest_point
			H[:,curr_idx] = (torch.norm(closest_point, p=2, dim=1) - self.param.r_agent)/(self.param.r_comm - self.param.r_agent)
			curr_idx += 1

		return P,H 

	def torch_fdbk_si(self,x,P,H):
		grad_phi = self.torch_get_grad_phi(x,P,H)
		b = -self.param.kp*grad_phi 
		return b

	def torch_scale(self,action,max_action):
		action_norm = action.norm(p=2,dim=1)
		index = action_norm > 0
		scale = torch.ones(action.shape[0],device=self.device)
		scale[index] = 1.0 / torch.clamp(action_norm[index]/max_action,min=1)
		action = torch.mul(scale.unsqueeze(1), action)
		return action

	def torch_get_cf_si_2(self,x,pi,barrier_action,P,H):
		epsilon = self.param.epsilon
		adaptive_scaling = torch.ones((H.shape[0],1),device=self.device) - epsilon
		# print('H',H)
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]

			grad_phi = self.torch_get_grad_phi(x,P,H)  
			A1 = self.param.kp * torch.pow(torch.norm(grad_phi,p=2,dim=1),2).unsqueeze(1)
			A2 = torch.bmm( grad_phi.unsqueeze(1), pi.unsqueeze(2)).squeeze(2)

			idx = minH < self.param.Delta_R / (self.param.r_comm - self.param.r_agent)
			hidx = A2 > 0
			adaptive_scaling[idx] = torch.min(\
				torch.mul(A1[idx],torch.pow(A1[idx] + torch.abs(A2[idx]),-1)),torch.ones(1,device=self.device) - epsilon)
			# adaptive_scaling[idx] = torch.min(\
			# 	torch.mul(A1[idx],torch.pow(A1[idx] + torch.mul(A2[idx],hidx[idx]),-1)),torch.ones(1,device=self.device))			
		return adaptive_scaling

	def torch_fdbk_di(self,x,P,H):
		v = -1*x[:,3:5]
		grad_phi = self.torch_get_grad_phi(x,P,H)
		grad_phi_dot = self.torch_get_grad_phi_dot(x,P,H)
		b = -self.param.kv*(v + self.param.kp*grad_phi) - self.param.kp*grad_phi_dot - self.param.kp*grad_phi
		return b

	def torch_get_cf_di_2(self,x,pi,b,P,H):
		epsilon = self.param.epsilon
		cf_alpha = torch.ones((H.shape[0],1),device=self.device) - epsilon
		# print('H',H)
		if not H.nelement() == 0:
			minH = torch.min(H,dim=1)[0]

			grad_phi = self.torch_get_grad_phi(x,P,H) 
			grad_phi_dot = self.torch_get_grad_phi_dot(x,P,H) 
			v = -x[:,3:5]
			vmk = v + self.param.kp * grad_phi 
			A1 = self.param.kp**2 * torch.pow(torch.norm(grad_phi,p=2,dim=1,keepdim=True),2) + \
				self.param.kv * torch.pow(torch.norm(vmk,p=2,dim=1,keepdim=True),2)
			A2 = (torch.bmm(vmk.unsqueeze(1), (pi + self.param.kp*grad_phi_dot).unsqueeze(2)) + \
				self.param.kp * torch.bmm(grad_phi.unsqueeze(1), v.unsqueeze(2))).squeeze(2)

			idx = minH < self.param.Delta_R / (self.param.r_comm - self.param.r_agent)
			cf_alpha[idx] = torch.min(\
				torch.mul(A1[idx],torch.pow(A1[idx] + torch.abs(A2[idx]),-1)),torch.ones(1,device=self.device)-epsilon) 

		return cf_alpha 

	def torch_get_grad_phi(self,x,P,H):
		grad_phi = torch.zeros((len(x),self.dim_action),device=self.device)
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			pj = P[:,j,:]
			hj = H[:,j]
			normp = torch.norm(pj,p=2,dim=1)
			denom = torch.mul(normp,hj)
			idx = normp > 0 
			grad_phi[idx] += torch.mul(pj[idx].transpose(0,1),torch.pow(denom[idx],-1)).transpose(0,1) \
				/(self.param.r_comm - self.param.r_agent) 
		return grad_phi

	def torch_get_grad_phi_dot(self,x,P,H):
		bs = x.shape[0]
		grad_phi_dot = torch.zeros((bs,1,2),device=self.device)

		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			normP = torch.norm(P[:,j,:],p=2,dim=1).unsqueeze(1)
			if j < self.get_num_neighbors(x):
				idx = self.get_agent_vel_idx_j(x,j)
			else:
				idx = np.array([3,4],dtype=int)

			v_rel = x[:,idx].unsqueeze(2)
			# print('v_rel',v_rel)

			p_rel = P[:,j,:].unsqueeze(1)
			normp = torch.norm(p_rel,p=2,dim=2) # bsx1

			f1 = p_rel
			f2 = torch.pow(normp,-1)
			f3 = torch.pow(normp-self.param.r_agent,-1)

			f1dot = torch.transpose(v_rel,1,2)
			f2dot = -torch.mul(torch.bmm(p_rel,v_rel).squeeze(2),torch.pow(normp,-3))
			f3dot = -torch.mul(torch.bmm(p_rel,v_rel).squeeze(2),torch.mul(torch.pow(normp,-1),torch.pow(normp-self.param.r_agent,-2)))

			idx = (normp > 0).squeeze()

			grad_phi_dot[idx] += \
				torch.mul(f1dot[idx],torch.mul(f2[idx],f3[idx]).unsqueeze(2)) + \
				torch.mul(f1[idx],torch.mul(f2dot[idx],f3[idx]).unsqueeze(2)) + \
				torch.mul(f1[idx],torch.mul(f2[idx],f3dot[idx]).unsqueeze(2))

		return grad_phi_dot.squeeze(1) 

	# numpy 
	def numpy_get_relative_positions_and_safety_functions(self,x):
		
		nd = x.shape[0] # number of data points in batch 
		nn = self.get_num_neighbors(x)
		no = self.get_num_obstacles(x) 

		P = np.zeros((nd,nn+no,2)) # pj - pi 
		H = np.zeros((nd,nn+no)) 
		curr_idx = 0

		for j in range(nn):
			idx = self.get_agent_pos_idx_j(x,j)
			P[:,curr_idx,:] = x[:,idx] * (1 - self.param.r_agent / np.linalg.norm(x[:,idx]))
			H[:,curr_idx] = (np.linalg.norm(P[:,curr_idx,:]) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
			curr_idx += 1 

		for j in range(no):
			idx = self.get_obstacle_idx_j(x,j)
			closest_point = min_point_circle_rectangle(
				np.zeros(2), 
				self.param.r_agent,
				x[:,idx] - np.array([0.5,0.5]), 
				x[:,idx] + np.array([0.5,0.5]))
			P[:,curr_idx,:] = closest_point
			H[:,curr_idx] = (np.linalg.norm(closest_point) - self.param.r_agent)/(self.param.r_obs_sense-self.param.r_agent)
			curr_idx += 1
		return P,H 

	def numpy_fdbk_si(self,x,P,H):
		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
		b = -self.param.kp*grad_phi
		return b

	def numpy_scale(self,action,max_action):
		norm_action = np.linalg.norm(action)
		if norm_action > 0:
			alpha = max_action/norm_action
			alpha = np.min((alpha,1))
			action = action*alpha
		return action

	def numpy_get_cf_si_2(self,x,P,H,pi,b):
		epsilon = self.param.epsilon
		adaptive_scaling = 1 - epsilon
		if not H.size == 0 and np.min(H) < self.param.Delta_R / (self.param.r_comm - self.param.r_agent):
			grad_phi = self.numpy_get_grad_phi(x,P,H) # 1x2
			A1 = self.param.kp*np.dot(grad_phi, grad_phi.T)
			A2 = np.dot(grad_phi,pi.T)
			adaptive_scaling = np.min((A1/(A1 + np.abs(A2)),1 - epsilon))
			# print(adaptive_scaling)
			# adaptive_scaling = np.min((A1/(A1 + np.heaviside(A2,1/2)*A2),1))
		return adaptive_scaling	

	def numpy_fdbk_di(self,x,P,H):
		v = -1*x[0,3:5]
		grad_phi = self.numpy_get_grad_phi(x,P,H) # in 1x2
		grad_phi_dot = self.numpy_get_grad_phi_dot(x,P,H)
		b = -1*self.param.kv*(v + self.param.kp*grad_phi) - self.param.kp*grad_phi_dot - self.param.kp*grad_phi
		return b 
		
	def numpy_get_cf_di_2(self,x,P,H,pi,b):
		epsilon = self.param.epsilon
		adaptive_scaling = 1 - epsilon
		if not H.size == 0 and np.min(H) < self.param.Delta_R / (self.param.r_comm - self.param.r_agent):
			grad_phi = self.numpy_get_grad_phi(x,P,H) # 1x2
			grad_phi_dot = self.numpy_get_grad_phi_dot(x,P,H) # 1x2
			v = -x[:,3:5]
			vmk = v + self.param.kp * grad_phi
			A1 = self.param.kp**2 * np.dot(grad_phi,grad_phi.T) + self.param.kv * np.dot( vmk,vmk.T)
			A2 = np.dot(vmk, (pi + self.param.kp*grad_phi_dot).T) + self.param.kp * np.dot(grad_phi, v.T)

			adaptive_scaling = np.min((A1/(A1 + np.abs(A2)),1-epsilon))
			
		return adaptive_scaling	

	def numpy_get_grad_phi_dot(self,x,P,H):
		grad_phi_dot = np.zeros((1,2))
		
		for j in range(self.get_num_neighbors(x)+self.get_num_obstacles(x)):
			
			if j < self.get_num_neighbors(x):
				idx = self.get_agent_vel_idx_j(x,j)
			else:
				idx = np.array([3,4],dtype=int)

			v_rel = x[:,idx] # 1x2
			p_rel = P[:,j,:] # 1x2 
			normp = np.linalg.norm(p_rel)
			if normp > 0:
				f1 = p_rel
				f2 = 1/normp
				f3 = 1/(normp-self.param.r_agent)
				f1dot = v_rel
				f2dot = -1/(normp**3)*np.dot(p_rel,v_rel.T)
				f3dot = -1/(normp*(normp-self.param.r_agent)**2)*np.dot(p_rel,v_rel.T)
				grad_phi_dot += f1dot*f2*f3 + f1*f2dot*f3 + f1*f2*f3dot

		return grad_phi_dot

	def numpy_get_grad_phi(self,x,P,H):
		grad_phi = np.zeros((len(x),self.dim_action))
		for j in range(self.get_num_neighbors(x) + self.get_num_obstacles(x)):
			grad_phi += self.numpy_get_grad_phi_contribution(P[:,j,:],H[:,j])
		return grad_phi		

	def numpy_get_grad_phi_contribution(self,P,H):
		normp = np.linalg.norm(P)
		grad_phi_ji = 0.
		if normp > 0:
			grad_phi_ji = P/(H*normp)/(self.param.r_comm - self.param.r_agent)
		return grad_phi_ji

	# helper fnc	
	def get_num_neighbors(self,x):
		return int(x[0,0])

	def get_num_obstacles(self,x):
		nn = self.get_num_neighbors(x)
		return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

	def get_agent_pos_idx_j(self,x,j):
		idx = 1+self.dim_state + self.dim_neighbor*j+np.arange(0,2,dtype=int)
		return idx

	def get_agent_vel_idx_j(self,x,j):
		idx = 2+1+self.dim_state + self.dim_neighbor*j+np.arange(0,2,dtype=int)
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