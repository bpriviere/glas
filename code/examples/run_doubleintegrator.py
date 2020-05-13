# # We use process parallelism, so multi-threading tends to hurt our performance
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

from param import Param
# TEMP
from run import run, parse_args
from sim import run_sim
from systems.doubleintegrator import DoubleIntegrator
from other_policy import APF, Empty_Net_wAPF, ZeroPolicy, GoToGoalPolicy

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

class DoubleIntegratorParam(Param):
	def __init__(self):
		super().__init__()
		self.env_name = 'DoubleIntegrator'
		self.env_case = None

		self.preprocessed_data_dir = 'data/preprocessed_data/'

		# flags
		self.pomdp_on = True
		self.single_agent_sim = False
		self.multi_agent_sim = True
		self.il_state_loss_on = False
		self.sim_render_on = False

		# orca param
		self.n_agents = 4
		self.r_comm = 3. #0.5
		self.r_obs_sense = 3.
		self.r_agent = 0.15 #0.2
		self.r_obstacle = 0.5
		self.v_max = 0.5
		self.a_max = 2.0 #.0 #2.0 # 7.5
		# self.v_max = 100 
		# self.a_max = 100
		self.v_min = -1*self.v_max
		self.a_min = -1*self.a_max

		# sim 
		self.sim_t0 = 0
		self.sim_tf = 5 #2*self.sim_dt # 0.1
		self.sim_dt = 0.5
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		self.max_neighbors = 6
		self.max_obstacles = 6
				
		self.safety = "cf_di_2" # potential, fdbk_di, cf_di, cf_di_2
		self.rollout_batch_on = True
		self.default_instance = "map_8by8_obst12_agents8_ex0009.yaml"
		self.current_model = 'il_current.pt'

		if self.safety == "fdbk_di":
			self.pi_max = 0.015 # 0.25
			self.kp = 0.005 # 0.002
			self.kv = 0.050 # 0.025
			self.cbf_kp = 0.5
			self.cbf_kd = 2.0

		elif self.safety == "cf_di": # 'working di' parameters
			self.pi_max = 2.0 # 0.05 
			self.sigmoid_scale = 5.0 #0.1 # 0.5 
			self.kp = 0.01 # 0.01 
			self.kv = 1.0 # 2.0 
			self.cbf_kp = 0.5 # 0.5 
			self.cbf_kd = 2.0 # 2.0 

		elif self.safety == "cf_di_2": # 'working di 2' parameters
			self.pi_max = 1.5 # 0.05 
			self.kp = 0.025 # 0.01 
			self.kv = 1.0 # 2.0 
			self.cbf_kp = 0.035 # 0.5
			self.cbf_kd = 0.5 # 2.0			 

		self.Delta_R = 2*(0.5*0.05 + 0.5**2/(2*2.0))
		self.epsilon = 0.01

		# obsolete parameters 
		self.b_gamma = .05 
		self.b_eps = 100
		self.b_exph = 1.0 
		self.D_robot = 1.*(self.r_agent+self.r_agent)
		self.D_obstacle = 1.*(self.r_agent + self.r_obstacle)
		self.circle_obstacles_on = True # square obstacles batch not implemented		

		# IL
		self.il_load_loader_on = False
		self.training_time_downsample = 50 #10
		self.il_train_model_fn = '../models/doubleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.85
		self.il_batch_size = 4096*8
		self.il_n_epoch = 5
		self.il_lr = 1e-3
		self.il_wd = 0 #0.0002
		self.il_n_data = None # 100000 # 100000000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Barrier' # 'Empty','Barrier',
		self.il_pretrain_weights_fn = None # None or path to *.tar file
		
		self.datadict = dict()
		self.datadict["8"] = 10000 #10000000 #750000 #self.il_n_data
		self.il_obst_case = 6
		self.controller_learning_module = 'DeepSet' #

		# adaptive dataset parameters
		self.adaptive_dataset_on = False
		self.ad_n = 100 # n number of rollouts
		self.ad_n_data_per_rollout = 100000 # repeat rollout until at least this amount of data was added
		self.ad_l = 2 # l prev observations 
		self.ad_k = 20 # k closest 
		self.ad_n_epoch = 10
		self.ad_n_data = 2000000
		self.ad_dl = 10 # every . timesteps  
		self.ad_train_model_fn = '../models/doubleintegrator/ad_current.pt'

		# Sim
		self.sim_rl_model_fn = '../models/doubleintegrator/rl_current.pt'
		self.sim_il_model_fn = '../models/doubleintegrator/il_current.pt'

		# plots
		self.vector_plot_dx = 0.25 		

		# self.ad_tf = 25 #25
		# self.ad_dt = 0.0
		# self.ad_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)

		# 
		# self.il_empty_model_fn = '../models/singleintegrator/empty.pt'
		# self.il_barrier_model_fn = '../models/singleintegrator/barrier.pt'
		# self.il_adaptive_model_fn = '../models/singleintegrator/adaptive.pt'

		# learning hyperparameters
		n,m,h,l,p = 4,2,64,16,16 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(4,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_phi_obs_network_architecture = nn.ModuleList([
			nn.Linear(4,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_rho_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_rho_obs_network_architecture = nn.ModuleList([
			nn.Linear(l,h),
			nn.Linear(h,h),
			nn.Linear(h,p)])

		self.il_psi_network_architecture = nn.ModuleList([
			nn.Linear(2*p+4,h),
			nn.Linear(h,h),
			nn.Linear(h,m)])

		self.il_network_activation = relu

		# plots
		self.vector_plot_dx = 0.3


def load_instance(param, env, instance):

	if False:
		# exp3: mean motion planning 
		# n_agents = 8
		# nd2 = int(n_agents/2)
		# start = np.zeros((n_agents,4))
		# start[0:nd2,0:2] = np.array([
		# 	[-2.,0.],
		# 	[-2.,1.],
		# 	[-1.,1.],
		# 	[-1.,-1.2],
		# 	])
		# start[nd2:,0:2] = -start[0:nd2,0:2]
		# goal = -start
		# obstacles = np.array([
		# 	[0,0.0],
		# 	[0,1.75],
		# 	[0,-1.75],
		# 	])	

		# num_agents = 12
		# r = 2.0
		# theta = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
		# start = np.zeros((num_agents,4))
		# start[:,0] = r * np.cos(theta)
		# start[:,1] = r * np.sin(theta)
		# goal = -start

		# # exp3: go to ring from grid
		# # n_x = 4
		# # n_y = 3
		# # l_x = (n_x-1)/4
		# # l_y = (n_y-1)/4
		# # temp = np.meshgrid(np.linspace(-l_x,l_x,n_x),np.linspace(-l_y,l_y,n_y))
		# # start[:,0] = temp[0].flatten()
		# # start[:,1] = temp[1].flatten()

		# obstacles = np.array([
		# 	# [0,0.0]
		# 	])	

		# 
		start = np.array([
			[4.,4.,0,0],
			[2,2,0,0],
			])
		goal = np.array([
			[2,2,0,0],
			[4,4,0,0],
			],dtype=np.float32)
		obstacles = np.array([
			[3., 3.],
			# [ -0.1, 1.12]
		])

		obstacles -= [0.5,0.5]
		InitialState = namedtuple('InitialState', ['start', 'goal'])
		s0 = InitialState._make((start.flatten(), goal.flatten()))
		param.n_agents = start.shape[0]
		env.reset_param(param)
		env.obstacles = obstacles
		return s0


	import yaml
	if instance:
		with open(instance) as map_file:
			map_data = yaml.load(map_file,Loader=yaml.SafeLoader)
	else:
		# default
		# instance = "map_8by8_obst6_agents64_ex0006.yaml"
		# instance = "map_8by8_obst6_agents32_ex0005.yaml"
		# instance = "map_8by8_obst6_agents16_ex0003.yaml"
		instance = "head_test.yaml"
		# with open("../results/singleintegrator/instances/{}".format(instance)) as map_file:
		with open("../results/singleintegrator/instances/{}".format(param.default_instance)) as map_file:
		# test map test dataset
			map_data = yaml.load(map_file)

	s = []
	g = []
	for agent in map_data["agents"]:
		s.extend([agent["start"][0] + 0.5, agent["start"][1] + 0.5])
		s.extend([0,0])
		g.extend([agent["goal"][0] + 0.5, agent["goal"][1] + 0.5])
		g.extend([0,0])

	InitialState = namedtuple('InitialState', ['start', 'goal'])
	s0 = InitialState._make((np.array(s), np.array(g)))

	param.n_agents = len(map_data["agents"])
	env.reset_param(param)

	env.obstacles = map_data["map"]["obstacles"]
	for x in range(-1,map_data["map"]["dimensions"][0]+1):
		env.obstacles.append([x,-1])
		env.obstacles.append([x,map_data["map"]["dimensions"][1]])
	for y in range(map_data["map"]["dimensions"][0]):
		env.obstacles.append([-1,y])
		env.obstacles.append([map_data["map"]["dimensions"][0],y])

	return s0


def run_batch(param, env, instance, controllers):
	torch.set_num_threads(1)
	s0 = load_instance(param, env, instance)
	for name, controller in controllers.items():
		print("Running simulation with " + name)

		states, observations, actions, step = run_sim(param, env, controller, s0, name=instance)
		# print(states[0:step].shape)
		# print(param.sim_times[0:step].shape)
		# exit()
		result = np.hstack((param.sim_times[0:step].reshape(-1,1), states[0:step]))
		# store in binary format
		basename = os.path.splitext(os.path.basename(instance))[0]
		folder_name = "../results/doubleintegrator/{}".format(name)
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)

		output_file = "{}/{}.npy".format(folder_name, basename)
		with open(output_file, "wb") as f:
			np.save(f, result.astype(np.float32), allow_pickle=False)

if __name__ == '__main__':

	args = parse_args()
	param = DoubleIntegratorParam()
	env = DoubleIntegrator(param)

	if args.il:
		run(param, env, None, None, args)
		exit()

	controllers = {
		# 'emptywapf': Empty_Net_wAPF(param,env,torch.load('../results/doubleintegrator/exp1Empty_0/il_current.pt')),
		# 'e2e':torch.load('../results/doubleintegrator/exp1Barrier_0/il_current.pt'),
		# 'empty':torch.load('../results/doubleintegrator/exp1Empty_0/il_current.pt'),

		# 'current':torch.load(param.il_train_model_fn),
		# 'current_wapf': Empty_Net_wAPF(param,env,torch.load(param.il_train_model_fn)),
		# 'gg': GoToGoalPolicy(param,env),
		'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env)),
		# 'zero': Empty_Net_wAPF(param,env,ZeroPolicy(env))
	}

	s0 = load_instance(param, env, args.instance)

	if args.batch:
		if args.controller:
			controllers = dict()
			for ctrl in args.controller:
				name,kind,path = ctrl.split(',')
				if kind == "EmptyAPF":
					controllers[name] = Empty_Net_wAPF(param,env,torch.load(path))
				elif kind == "torch":
					controllers[name] = torch.load(path)
				elif kind == "apf":
					controllers[name] = Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env))
				else:
					print("ERROR unknown ctrl kind", kind)
					exit()

		if args.Rsense:
			param.r_comm = args.Rsense
			param.r_obs_sense = args.Rsense
		if args.maxNeighbors:
			param.max_neighbors = args.maxNeighbors
			param.max_obstacles = args.maxNeighbors
		env.reset_param(param)

		run_batch(param, env, args.instance, controllers)

	elif args.export:
		# model = torch.load('/home/whoenig/pCloudDrive/caltech/neural_pid_results/doubleintegrator/il_current.pt')
		# change path 
		# model = torch.load('/home/ben/pCloudDrive/arcl/neural_pid/results/neural_pid_results/doubleintegrator/il_current.pt')
		# model = torch.load('/home/ben/projects/caltech/neural-pid/results/doubleintegrator/exp1Empty_0/il_current.pt')
		model = torch.load('/home/ben/projects/caltech/neural-pid/results/doubleintegrator/exp1Barrier_0/il_current.pt')
		# model = torch.load('/home/ben/pCloudDrive/arcl/neural_pid/results/neural_pid_results/di_investigation/40m_cf_2/exp1Empty_0/il_current.pt')
		# model = torch.load('/home/ben/pCloudDrive/arcl/neural_pid/results/neural_pid_results/di_investigation/40m_comp_filter/empty.pt')
		model.export_to_onnx("IL")

	else:
		run(param, env, controllers, s0, args)

