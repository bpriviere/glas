
# Creating OpenAI gym Envs 

# standard package
from gym import Env
from collections import namedtuple
import numpy as np 
import torch
from scipy import spatial
import os
import yaml

# my package
import utilities
from utilities import rot_mat_2d, not_batch_is_collision_circle_rectangle
from scipy.linalg import block_diag

class Agent:
	def __init__(self,i):
		self.i = i 
		self.s = None
		self.p = None
		self.v = None
		self.s_g = None

class SingleIntegrator(Env):

	def __init__(self, param):
		self.reset_param(param)

	def reset_param(self, param):
		# init
		self.times = param.sim_times
		self.state = None
		self.time_step = None

		self.total_time = param.sim_times[-1]
		self.dt = param.sim_times[1] - param.sim_times[0]

		self.n_agents = param.n_agents
		self.state_dim_per_agent = 2
		self.action_dim_per_agent = 2
		self.r_agent = param.r_agent
		self.r_obstacle = param.r_obstacle
		self.r_obs_sense = param.r_obs_sense
		self.r_comm = param.r_comm 

		# control lim
		self.a_min = param.a_min
		self.a_max = param.a_max

		# default parameters [SI units]
		self.n = self.state_dim_per_agent*self.n_agents
		self.m = self.action_dim_per_agent*self.n_agents

		# init agents
		self.agents = []
		for i in range(self.n_agents):
			self.agents.append(Agent(i))

		# environment 
		self.init_state_mean = 0.
		self.init_state_var = 10.

		self.states_name = [
			'x-Position [m]',
			'y-Position [m]',
			]
		self.actions_name = [
			'x-Velocity [m/s]',
			'y-Velocity [m/s]'
			]

		self.param = param
		self.max_reward = 0 

		self.obstacles = []


	def render(self):
		pass

	def step(self, a, compute_reward = True):
		self.s = self.next_state(self.s, a)
		d = self.done()
		if compute_reward:
			r = self.reward()
		else:
			r = 0
		self.time_step += 1
		return self.s, r, d, {}

	def done(self):
		
		for agent in self.agents:
			if not np.linalg.norm(agent.s - agent.s_g) < 0.05:
				return False
		return True

	def observe(self):
		observations = []
		oa_pairs = []
		for agent_i in self.agents:
			p_i = agent_i.p
			s_i = agent_i.s
			relative_goal = torch.Tensor(agent_i.s_g - s_i)
			
			time_to_goal = self.total_time - self.time_step * self.dt

			# query visible neighbors
			_, neighbor_idx = self.kd_tree_neighbors.query(p_i,
				k=self.param.max_neighbors+1,
				distance_upper_bound=self.param.r_comm)
			if type(neighbor_idx) is not np.ndarray:
				neighbor_idx = [neighbor_idx]
			relative_neighbors = []
			for k in neighbor_idx[1:]: # skip first entry (self)
				if k < self.positions.shape[0]:
					relative_neighbors.append(self.agents[k].s - s_i)
				else:
					break

			# query visible obstacles
			if self.param.max_obstacles > 0:
				_, obst_idx = self.kd_tree_obstacles.query(p_i,
					k=self.param.max_obstacles,
					distance_upper_bound=self.param.r_obs_sense)
				if type(obst_idx) is not np.ndarray:
					obst_idx = [obst_idx]
			else:
				obst_idx = []
			relative_obstacles = []
			for k in obst_idx:
				if k < self.obstacles_np.shape[0]:
					relative_obstacles.append(self.obstacles_np[k,:] - p_i)
					# closest = utilities.min_point_circle_rectangle(
					# 	p_i,
					# 	self.param.r_agent,
					# 	self.obstacles_np[k,:] - np.array([0.5,0.5]),
					# 	self.obstacles_np[k,:] + np.array([0.5,0.5]))
					# relative_obstacles.append(closest - p_i)
				else:
					break

			# convert to numpy array format
			num_neighbors = len(relative_neighbors)
			num_obstacles = len(relative_obstacles)
			obs_array = np.zeros(3+2*num_neighbors+2*num_obstacles, dtype=np.float32)
			obs_array[0] = num_neighbors
			idx = 1
			obs_array[idx:idx+2] = relative_goal
			idx += 2
			# obs_array[4] = observation_i.time_to_goal
			for i in range(num_neighbors):
				obs_array[idx:idx+2] = relative_neighbors[i]
				idx += 2
			for i in range(num_obstacles):
				obs_array[idx:idx+2] = relative_obstacles[i]
				idx += 2

			oa_pairs.append((obs_array,np.zeros((self.action_dim_per_agent))))
			observations.append(obs_array)
			# observations.append(observation_i)

		transformed_oa_pairs, _ = self.preprocess_transformation(oa_pairs)
		observations = [o for (o,a) in transformed_oa_pairs]
		return observations

	def reward(self):
		# check with respect to other agents
		results = self.kd_tree_neighbors.query_pairs(2*self.r_agent)
		if len(results) > 0:
			for result in results:
				p_i = self.agents[result[0]].p
				p_j = self.agents[result[1]].p
				dist = np.linalg.norm(p_i-p_j)
				print('   a2a dist {} at time {} a {} and a {}'.format(dist,self.times[self.time_step], self.agents[result[0]].i, self.agents[result[1]].i))
				# exit()
			return -1

		# check with respect to obstacles
		# results = self.kd_tree_obstacles.query_ball_point(self.positions, self.r_agent + self.r_obstacle)
		# for r in results:
		# 	if len(r) > 0:
		# 		return -1

		for agent in self.agents:
			for o in self.obstacles:
				coll, dist = not_batch_is_collision_circle_rectangle(np.array(agent.p), self.param.r_agent, np.array(o), np.array(o) + np.array([1.0,1.0]))
				inc = np.count_nonzero(coll)
				if inc > 0:
					print('   a2o dist {} at time {} a {}'.format(dist,self.times[self.time_step], agent.i))
					return -inc 

		return 0


	def reset(self, initial_state=None):
		self.time_step = 0

		# initialize agents
		if initial_state is None:
			initial_state = np.zeros((self.n))
			for agent_i in self.agents:
				agent_i.s = self.find_collision_free_state('initial')
				# agent_i.s_g = self.find_collision_free_state('goal')
				agent_i.s_g = -agent_i.s
				idx = self.agent_idx_to_state_idx(agent_i.i) + \
					np.arange(0,self.state_dim_per_agent)
				initial_state[idx] = agent_i.s
			self.s = initial_state
		else:
			# print(initial_state)

			# this will update agents later in 'update_agents'
			self.s = initial_state.start
			
			# make agent list correct
			self.n_agents = int(len(self.s)/self.state_dim_per_agent)
			self.n = self.state_dim_per_agent*self.n_agents
			self.m = self.action_dim_per_agent*self.n_agents			
			self.agents = []
			for i in range(self.n_agents):
				self.agents.append(Agent(i))

			# update goal position
			for agent in self.agents:
				idx = self.agent_idx_to_state_idx(agent.i) + \
					np.arange(0,self.state_dim_per_agent)
				agent.s_g = initial_state.goal[idx]
				

		self.obstacles_np = np.array([np.array(o) + np.array([0.5,0.5]) for o in self.obstacles])
		self.kd_tree_obstacles = spatial.KDTree(self.obstacles_np)

		self.update_agents(self.s)
		return np.copy(self.s)


	def find_collision_free_state(self,config):
		collision = True
		count = 0 
		while collision:
			count += 1
			collision = False
			s = self.init_state_mean + \
					self.init_state_var*np.random.uniform(size=(self.state_dim_per_agent))
			for agent_j in self.agents:
				if agent_j.s is not None and agent_j.s_g is not None:
					if config == 'initial': 
						dist = np.linalg.norm(agent_j.s[0:2] - s[0:2])
					elif config == 'goal':
						dist = np.linalg.norm(agent_j.s_g[0:2] - s[0:2])
					if dist < 2*self.r_agent:
						collision = True
						break

			if count > 1000:
				print('Infeasible initial conditions')
				exit()

		return s 


	def next_state(self,s,a):

		sp1 = np.zeros((self.n))
		dt = self.times[self.time_step+1]-self.times[self.time_step]

		# single integrator
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			p_idx = np.arange(idx,idx+2)
			sp1[p_idx] = self.s[p_idx] + a[agent_i.i,:]*dt
			agent_i.v = a[agent_i.i,:]
			# sp1[v_idx] = np.clip(a[agent_i.i,:],self.a_max,self.a_min)

		self.update_agents(sp1)
		return sp1
		

	def update_agents(self,s):
		for agent_i in self.agents:
			idx = self.agent_idx_to_state_idx(agent_i.i)
			agent_i.p = s[idx:idx+2]
			# agent_i.v = s[idx+2:idx+4]
			agent_i.s = agent_i.p

		self.positions = np.array([agent_i.p for agent_i in self.agents])
		self.kd_tree_neighbors = spatial.KDTree(self.positions)

	def agent_idx_to_state_idx(self,i):
		return self.state_dim_per_agent*i 


	def load_dataset_action_loss(self, filename):
		data = np.load(filename)

		# load map
		instance = os.path.splitext(os.path.basename(filename))[0]

		filename_map = "{}/../instances/{}.yaml".format(os.path.dirname(filename), instance)
		with open(filename_map) as map_file:
			map_data = yaml.load(map_file, Loader=yaml.SafeLoader)
		obstacles = []
		for o in map_data["map"]["obstacles"]:
			obstacles.append(np.array(o) + np.array([0.5,0.5]))

		for x in range(-1,map_data["map"]["dimensions"][0]+1):
			obstacles.append(np.array([x,-1]) + np.array([0.5,0.5]))
			obstacles.append(np.array([x,map_data["map"]["dimensions"][1]]) + np.array([0.5,0.5]))
		for y in range(map_data["map"]["dimensions"][0]):
			obstacles.append(np.array([-1,y]) + np.array([0.5,0.5]))
			obstacles.append(np.array([map_data["map"]["dimensions"][0],y]) + np.array([0.5,0.5]))

		obstacles = np.array(obstacles)
		kd_tree_obstacles = spatial.KDTree(obstacles)

		# find goal times
		goal_idxs = []
		for i, agent in enumerate(map_data["agents"]):
			goal = np.array([0.5,0.5]) + np.array(agent["goal"])
			distances = np.linalg.norm(data[:,(i*4+1):(i*4+3)] - goal, axis=1)
			goalIdx = np.argwhere(distances > 0.1)
			if len(goalIdx) == 0:
				goalIdx = np.array([0])
			lastIdx = np.max(goalIdx)
			if lastIdx < data.shape[0] - 1:
				goal_idxs.append(lastIdx)
			else:
				goal_idxs.append(data.shape[0] - 1)

		data = torch.from_numpy(data)

		num_agents = int((data.shape[1] - 1) / 4)
		dataset = []
		# Observation_Action_Pair = namedtuple('Observation_Action_Pair', ['observation', 'action']) 
		# Observation = namedtuple('Observation',['relative_goal','time_to_goal','relative_neighbors','relative_obstacles']) 
		for t in range(0,data.shape[0]-1):
			if t%self.param.training_time_downsample != 0:
				continue

			# build kd-tree
			positions = np.array([data[t,i*4+1:i*4+3].numpy() for i in range(num_agents)])
			kd_tree_neighbors = spatial.KDTree(positions)

			for i in range(num_agents):
				# skip datapoints where agents are just sitting at goal
				if t >= goal_idxs[i]:
					continue

				s_i = data[t,i*4+1:i*4+3]   # state i 
				# s_g = data[-1,i*4+1:i*4+5]  # goal state i 
				s_g = torch.Tensor(map_data["agents"][i]["goal"]) + torch.Tensor([0.5,0.5])
				# print(s_g, data[-1,i*4+1:i*4+5])
				relative_goal = s_g - s_i   # relative goal
				time_to_goal = data[-1,0] - data[t,0]

				# query visible neighbors
				_, neighbor_idx = kd_tree_neighbors.query(
					s_i[0:2].numpy(),
					k=self.param.max_neighbors+1,
					distance_upper_bound=self.param.r_comm)
				if type(neighbor_idx) is not np.ndarray:
					neighbor_idx = [neighbor_idx]
				relative_neighbors = []
				for k in neighbor_idx[1:]: # skip first entry (self)
					if k < positions.shape[0]:
						relative_neighbors.append(data[t,k*4+1:k*4+3] - s_i)
					else:
						break

				# query visible obstacles
				if self.param.max_obstacles > 0:
					_, obst_idx = kd_tree_obstacles.query(
						s_i[0:2].numpy(),
						k=self.param.max_obstacles,
						distance_upper_bound=self.param.r_obs_sense)
					if type(obst_idx) is not np.ndarray:
						obst_idx = [obst_idx]
				else:
					obst_idx = []
				relative_obstacles = []
				for k in obst_idx:
					if k < obstacles.shape[0]:
						relative_obstacles.append(obstacles[k,:] - s_i[0:2].numpy())
						# closest = utilities.min_point_circle_rectangle(
						# 	s_i[0:2].numpy(),
						# 	self.param.r_agent,
						# 	obstacles[k,:] - np.array([0.5,0.5]),
						# 	obstacles[k,:] + np.array([0.5,0.5]))
						# relative_obstacles.append(closest - s_i[0:2].numpy())
					else:
						break

				num_neighbors = len(relative_neighbors)
				num_obstacles = len(relative_obstacles)

				obs_array = np.empty(3+2*num_neighbors+2*num_obstacles+2, dtype=np.float32)
				obs_array[0] = num_neighbors
				idx = 1
				obs_array[idx:idx+2] = relative_goal
				idx += 2
				# obs_array[4] = data.observation.time_to_goal
				for k in range(num_neighbors):
					obs_array[idx:idx+2] = relative_neighbors[k]
					idx += 2
				for k in range(num_obstacles):
					obs_array[idx:idx+2] = relative_obstacles[k]
					idx += 2
				# action: velocity
				obs_array[idx:idx+2] = data[t+1, i*4+3:i*4+5]
				idx += 2

				transformed_oa_pairs, _ = self.preprocess_transformation([(obs_array[0:-2],obs_array[-2:])])

				dataset.append(np.hstack(transformed_oa_pairs[0]).flatten())

		return dataset


	def preprocess_transformation(self, dataset_batches):
		# input: 
		# 	- list of tuple of (observation, actions) pairs, numpy/pytorch supported
		# output: 
		# 	- list of tuple of (observation, actions) pairs, numpy arrays
		# 	- list of transformations 

		# TEMP 
		obstacleDist = self.param.r_obs_sense
		transformed_dataset_batches = []
		transformations_batches = []	
		for (dataset, classification) in dataset_batches:

			# dataset = [#n, sg-si, {sj-si}, {so-si}]

			if isinstance(dataset,torch.Tensor):
				dataset = dataset.detach().numpy()
			if isinstance(classification,torch.Tensor):
				classification = classification.detach().numpy()
					
			if dataset.ndim == 1:
				dataset = np.reshape(dataset,(-1,len(dataset)))
			if classification.ndim == 1:
				classification = np.reshape(classification,(-1,len(classification)))

			num_neighbors = int(dataset[0,0]) #int((x.size()[1]-4)/4)
			num_obstacles = int((dataset.shape[1]-3-2*num_neighbors)/2)

			idx_goal = np.arange(1,3,dtype=int)

			transformed_dataset = np.empty(dataset.shape,dtype=np.float32)
			transformed_classification = np.empty(classification.shape,dtype=np.float32)
			transformations = np.empty((dataset.shape[0],2,2),dtype=np.float32)

			for k,row in enumerate(dataset):

				transformed_row = np.empty(row.shape,dtype=np.float32)
				transformed_row[0] = row[0]

				# get goal 
				# s_gi = sg - si 
				s_gi = row[idx_goal]

				# get transformation 
				th = 0
				# th = np.arctan2(s_gi[1],s_gi[0])
				
				R = rot_mat_2d(th)

				# conditional normalization of relative goal
				dist = np.linalg.norm(s_gi[0:2])
				if dist > obstacleDist:
					s_gi[0:2] = s_gi[0:2] / dist * obstacleDist

				# transform goal 
				transformed_row[idx_goal] = np.matmul(R,s_gi)

				# get neighbors
				# transform neighbors 
				for j in range(num_neighbors):
					idx = 1+2+j*2+np.arange(0,2,dtype=int)
					s_ji = row[idx] 
					transformed_row[idx] = np.matmul(R,s_ji)

				# get obstacles
				# transform neighbors 
				for j in range(num_obstacles):
					idx = 1+2+num_neighbors*2+j*2+np.arange(0,2,dtype=int)
					s_oi = row[idx] 
					transformed_row[idx] = np.matmul(R,s_oi)
				
				# transform action
				if classification is not None: 
					transformed_classification[k,:] = np.matmul(R,classification[k])
				transformed_dataset[k,:] = transformed_row
				transformations[k,:,:] = R

			transformed_dataset_batches.append((transformed_dataset,transformed_classification))
			transformations_batches.append(transformations)

		return transformed_dataset_batches, transformations_batches


	def visualize(self,states,dt):

		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		import time 

		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		for i in range(self.n_agents):
			vis["agent"+str(i)].set_object(g.Sphere(self.r_agent))

		while True:
			for state in states:
				for i in range(self.n_agents):
					idx = self.agent_idx_to_state_idx(i) + np.arange(0,2)
					pos = state[idx]
					vis["agent" + str(i)].set_transform(tf.translation_matrix([pos[0], pos[1], 0]))
				time.sleep(dt)

	def instance_to_initial_state(self,instance):

		InitialState = namedtuple('InitialState', ['start', 'goal'])
		s,g = [],[]
		for agent in instance["agents"]:
			s.extend([agent["start"][0] + 0.5, agent["start"][1] + 0.5])
			g.extend([agent["goal"][0] + 0.5, agent["goal"][1] + 0.5])
		s0 = InitialState._make((np.array(s), np.array(g)))

		self.obstacles = instance["map"]["obstacles"]
		for x in range(-1,instance["map"]["dimensions"][0]+1):
			self.obstacles.append([x,-1])
			self.obstacles.append([x,instance["map"]["dimensions"][1]])
		for y in range(instance["map"]["dimensions"][0]):
			self.obstacles.append([-1,y])
			self.obstacles.append([instance["map"]["dimensions"][0],y])
		return s0

