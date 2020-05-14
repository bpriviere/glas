# # We use process parallelism, so multi-threading tends to hurt our performance
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

# TEMP
from run import run, parse_args
from sim import run_sim
from systems.doubleintegrator import DoubleIntegrator
from other_policy import Empty_Net_wAPF, ZeroPolicy, GoToGoalPolicy

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

class DoubleIntegratorParam():
	def __init__(self):
		self.env_name = 'DoubleIntegrator'
		self.env_case = None

		# some path param
		self.preprocessed_data_dir = 'data/preprocessed_data/'
		self.default_instance = "map_8by8_obst6_agents8_ex0001.yaml"
		self.current_model = 'il_current.pt'

		# dont change these sim param (same as ORCA baseline)
		self.n_agents = 4
		self.r_comm = 3. 
		self.r_obs_sense = 3.
		self.r_agent = 0.15 
		self.r_obstacle = 0.5
		self.v_max = 0.5
		self.a_max = 2.0 
		self.v_min = -1*self.v_max
		self.a_min = -1*self.a_max

		# sim 
		self.sim_t0 = 0
		self.sim_tf = 5 
		self.sim_dt = 0.5
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# for batching/speed
		self.max_neighbors = 6
		self.max_obstacles = 6
		self.rollout_batch_on = True
				
		# safety parameters
		self.safety = "cf_di_2" 
		if self.safety == "cf_di_2": # 'working di 2' parameters
			self.pi_max = 1.5 # 0.05 
			self.kp = 0.025 # 0.01 
			self.kv = 1.0 # 2.0 
			self.cbf_kp = 0.035 # 0.5
			self.cbf_kd = 0.5 # 2.0			 
		self.Delta_R = 2*(0.5*0.05 + 0.5**2/(2*2.0))
		self.epsilon = 0.01

		# imitation learning param (IL)
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
		
		# dataset param
		# ex: only take 8 agent cases, stop after 10K points 
		self.datadict = dict()
		self.datadict["8"] = 10000 

		# plots
		self.vector_plot_dx = 0.25 

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

	import yaml
	if instance:
		with open(instance) as map_file:
			map_data = yaml.load(map_file,Loader=yaml.SafeLoader)
	else:
		# default
		with open("../results/doubleintegrator/instances/{}".format(param.default_instance)) as map_file:
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
		result = np.hstack((param.sim_times[0:step].reshape(-1,1), states[0:step]))
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
		'current':torch.load(param.il_train_model_fn),
		# 'current_wapf': Empty_Net_wAPF(param,env,torch.load(param.il_train_model_fn)),
		# 'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env)),
		# 'e2e' : torch.load('../results/doubleintegrator/exp1Barrier_0/il_current.pt'),
		# '2stage' : Empty_Net_wAPF(param,env,torch.load('../results/doubleintegrator/exp1Empty_0/il_current.pt')),
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
		model = torch.load('/home/ben/projects/caltech/neural-pid/results/doubleintegrator/exp1Barrier_0/il_current.pt')
		model.export_to_onnx("IL")

	else:
		run(param, env, controllers, s0, args)

