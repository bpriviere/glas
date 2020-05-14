# We use process parallelism, so multi-threading tends to hurt our performance
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from run import run, parse_args
from sim import run_sim
from systems.singleintegrator import SingleIntegrator
from other_policy import Empty_Net_wAPF 

# standard
from torch import nn, tanh, relu
import torch
import numpy as np
from collections import namedtuple
import os

class SingleIntegratorParam():
	def __init__(self):
		self.env_name = 'SingleIntegrator'
		self.env_case = None

		# some path param 
		self.preprocessed_data_dir = 'data/preprocessed_data/'
		self.default_instance = "map_8by8_obst12_agents8_ex0000.yaml"
		self.current_model = 'il_current.pt'

		# orca param
		self.n_agents = 1
		self.r_comm = 3 
		self.r_obs_sense = 3.
		self.r_agent = 0.15
		self.r_obstacle = 0.5

		# sim 
		self.sim_t0 = 0
		self.sim_tf = 5
		self.sim_dt = 0.5
		self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		self.sim_nt = len(self.sim_times)
		self.plots_fn = 'plots.pdf'

		# speed/batching
		self.rollout_batch_on = True
		self.max_neighbors = 6
		self.max_obstacles = 6

		# safety
		self.safety = "cf_si_2" 
		if self.safety == "cf_si_2":
			self.a_max = 0.5	
			self.pi_max = 0.8	
			self.kp = 1.5		
			self.cbf_kp = 0.5
			self.epsilon = 0.01
			pi_max_thresh = self.kp / (0.2 - self.r_agent) * 0.01 # 0.01 = epsilon
			print('pi_max_thresh = ',pi_max_thresh)
		self.Delta_R = 2*self.a_max*self.sim_dt
		self.a_min  = -self.a_max
		self.pi_min = -self.pi_max

		# imitation learning param (IL)
		self.il_load_loader_on = True
		self.training_time_downsample = 50
		self.il_train_model_fn = '../models/singleintegrator/il_current.pt'
		self.il_imitate_model_fn = '../models/singleintegrator/rl_current.pt'
		self.il_load_dataset_on = True
		self.il_test_train_ratio = 0.85
		self.il_batch_size = 4096*8
		self.il_n_epoch = 5
		self.il_lr = 1e-3
		self.il_wd = 0 #0.0001
		self.il_n_data = None # 100000 # 100000000
		self.il_log_interval = 1
		self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
		self.il_controller_class = 'Empty' # 'Empty','Barrier',
		self.il_pretrain_weights_fn = None # None or path to *.tar file
		
		# dataset param 
		# ex: only take 8 agent cases, stop after 10K points
		self.datadict = dict()
		self.datadict["8"] = 10000 # 100000000000 #10000000 #750000 #self.il_n_data

		# learning hyperparameters
		n,m,h,l,p = 2,2,64,16,16 # state dim, action dim, hidden layer, output phi, output rho
		self.il_phi_network_architecture = nn.ModuleList([
			nn.Linear(2,h),
			nn.Linear(h,h),
			nn.Linear(h,l)])

		self.il_phi_obs_network_architecture = nn.ModuleList([
			nn.Linear(2,h),
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
			nn.Linear(2*p+2,h),
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
		with open("../results/singleintegrator/instances/{}".format(param.default_instance)) as map_file:
			map_data = yaml.load(map_file)

	s = []
	g = []
	for agent in map_data["agents"]:
		s.extend([agent["start"][0] + 0.5, agent["start"][1] + 0.5])
		g.extend([agent["goal"][0] + 0.5, agent["goal"][1] + 0.5])

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
		states_and_actions = np.zeros((step, 4*param.n_agents), dtype=np.float32)
		states_and_actions[:,0::4] = states[:step,0::env.state_dim_per_agent]
		states_and_actions[:,1::4] = states[:step,1::env.state_dim_per_agent]
		states_and_actions[:,2::4] = actions[:step,0::env.action_dim_per_agent]
		states_and_actions[:,3::4] = actions[:step,1::env.action_dim_per_agent]

		result = np.hstack((param.sim_times[0:step].reshape(-1,1), states_and_actions))

		basename = os.path.splitext(os.path.basename(instance))[0]
		folder_name = "../results/singleintegrator/{}".format(name)
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)
		output_file = "{}/{}.npy".format(folder_name, basename)
		with open(output_file, "wb") as f:
			np.save(f, result.astype(np.float32), allow_pickle=False)

if __name__ == '__main__':

	args = parse_args()
	param = SingleIntegratorParam()
	env = SingleIntegrator(param)

	if args.il:
		run(param, env, None, None, args)
		exit()

	controllers = {
		'current':torch.load(param.il_train_model_fn),
		# 'current_wapf': Empty_Net_wAPF(param,env,torch.load(param.il_train_model_fn)),
		# 'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env)),
		# 'e2e' : torch.load('../results/singleintegrator/exp1Barrier_0/il_current.pt'),
		# '2stage' : Empty_Net_wAPF(param,env,torch.load('../results/singleintegrator/exp1Empty_0/il_current.pt')),
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
		model = torch.load(param.il_train_model_fn)
		model.export_to_onnx("IL")

	else:
		torch.set_num_threads(1)
		run(param, env, controllers, s0, args)

