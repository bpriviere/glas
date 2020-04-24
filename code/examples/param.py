

import numpy as np 

class Param:
	def __init__(self):

		# reinforcement learning parameters
		# self.rl_gpu_on = False
		# self.rl_continuous_on = True
		# self.rl_log_interval = 20
		# self.rl_save_model_interval = np.Inf
		# self.rl_max_episodes = 50000
		# self.rl_train_model_fn = 'rl_model.pt'
		# self.rl_batch_size = 1000
		# self.rl_gamma = 0.98
		# self.rl_K_epoch = 5
		# self.rl_control_lim = 10
		# self.rl_card_A = 25
		# self.rl_discrete_action_space = np.linspace(\
		# 	-self.rl_control_lim,
		# 	 self.rl_control_lim,
		# 	 self.rl_card_A)
		# # ppo param
		# self.rl_lr = 5e-3
		# self.rl_lmbda = 0.95
		# self.rl_eps_clip = 0.2
		# # ddpg param
		# self.rl_lr_mu = 1e-4
		# self.rl_lr_q = 1e-3
		# self.rl_buffer_limit = 5e6
		# self.rl_action_std = 2
		# self.rl_max_action_perturb = 5
		# self.rl_tau = 0.995

		# # imitation learning parameters
		# self.il_lr = 5e-4
		# self.il_n_epoch = 50000 # number of epochs per batch 
		# self.il_batch_size = 2000 # number of data points per batch
		# self.il_n_data = 3000 # total number of data points 
		# self.il_log_interval = 1

		# # sim parameters
		# self.sim_t0 = 0
		# self.sim_tf = 500
		# self.sim_dt = 0.25
		# self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
		# self.sim_nt = len(self.sim_times)

		# # plots
		# self.plots_fn = 'plots.pdf'

		# # desired tracking trajectory
		# self.ref_trajectory = np.zeros((4,self.sim_nt))

		# # Planning
		# self.rrt_fn = 'rrt.csv'
		# self.scp_fn = 'scp.csv'
		# self.scp_pdf_fn = 'scp.pdf'
		pass 
param = Param()


# sys.path.insert(1, os.path.join(os.getcwd(),'learning'))
# sys.path.insert(1, os.path.join(os.getcwd(),'systems'))