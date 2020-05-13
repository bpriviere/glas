
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np 
import random 
import glob
import os
import shutil
import yaml
import utilities
import concurrent.futures
from itertools import repeat
import time

from numpy import array, zeros, Inf
from numpy.random import uniform,seed
from scipy import spatial
from torch.distributions import Categorical
from collections import namedtuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

from learning.empty_net import Empty_Net
from learning.barrier_net import Barrier_Net

from index import Index 

from genRandomInstanceDict import get_random_instance
import pprint

pp = pprint.PrettyPrinter(indent=4)

def rollout(model, env, param):
	
	keys = list(param.datadict.keys())
	p = np.array([param.datadict[k] for k in keys])
	p = p / np.sum(p)
	agent_case = int(np.random.choice(keys,p=p))

	instance = get_random_instance(agent_case,param.il_obst_case)
	initial_state = env.instance_to_initial_state(instance)

	observations = [] 
	env.reset(initial_state)
	for step, time in enumerate(param.sim_times[:-1]):

		observation = env.observe()
		action = model.policy(observation,env.transformations)
		next_state, _, done, _ = env.step(action, compute_reward = False)

		observations.append(observation)

		agents = env.bad_behavior(observation)
		past_l_observations = []
		for agent in agents:
			for past_l_observation in observations[-param.ad_l*param.ad_dl::param.ad_dl]: 
				past_l_observations.append(past_l_observation[agent.i])

		if len(agents) > 0:
			return past_l_observations

		if done: 
			break

	print('no bad behavior :)')
	return [] 

def get_dynamic_dataset(model, env, param,index):

	# inputs:
	#    - model
	#    - environment
	#    - param
	
	data = [] 
	print('rollout')

	while len(data) < param.ad_n_data_per_rollout:
		with concurrent.futures.ProcessPoolExecutor() as executor:
			for observation_i in executor.map(rollout, repeat(model,param.ad_n),repeat(env,param.ad_n),repeat(param,param.ad_n)):
			# for observation_i in executor.map(rollout, repeat(model,param.ad_n),env_lst,repeat(param,param.ad_n)):
				data.extend(index.query_lst(observation_i,param.ad_k))
				print('rollout data: ',len(data))

	# for i in range(param.ad_n):
	# 	observation_i = rollout(model, env, param)
	# 	data.extend(index.query_lst(observation_i,param.ad_k))


	print('end rollout')

	# print(len(data))
	# exit()
	return data 


def make_loader(
	env,
	dataset=None,
	n_data=None,
	shuffle=False,
	batch_size=None,
	name=None,
	device=None):

	def batch_loader(env, dataset, device):
		# break by observation size
		dataset_dict = dict()

		for data in dataset:
			num_neighbors = int(data[0])
			if env.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
				num_obstacles = int((data.shape[0] - 1 - env.state_dim_per_agent - num_neighbors*env.state_dim_per_agent - 2) / 2)
			elif env.param.env_name == 'SingleIntegratorVelSensing':
				num_obstacles = int((data.shape[0] - 1 - 2 - num_neighbors*4 - 2) / 2)

			key = (num_neighbors, num_obstacles)
			if key in dataset_dict:
				dataset_dict[key].append(data)
			else:
				dataset_dict[key] = [data]

		# Create actual batches
		loader = []
		for key, dataset_per_key in dataset_dict.items():
			num_neighbors, num_obstacles = key
			batch_x = []
			batch_y = []
			for data in dataset_per_key:
				batch_x.append(data[0:-2])
				batch_y.append(data[-2:])

			# store all the data for this nn/no-pair in a file
			batch_x = np.array(batch_x, dtype=np.float32)
			batch_y = np.array(batch_y, dtype=np.float32)
			batch_xy = np.hstack((batch_x, batch_y))

			print(name, " neighbors ", num_neighbors, " obstacles ", num_obstacles, " ex. ", batch_x.shape[0])

			with open("../preprocessed_data/batch_{}_nn{}_no{}.npy".format(name,num_neighbors, num_obstacles), "wb") as f:
				np.save(f, batch_xy, allow_pickle=False)

			# convert to torch
			batch_xy_torch = torch.from_numpy(batch_xy).float().to(device)

			# split data by batch size
			for idx in np.arange(0, batch_x.shape[0], batch_size):
				last_idx = min(idx + batch_size, batch_x.shape[0])
				# print("Batch of size ", last_idx - idx)
				x_data = batch_xy_torch[idx:last_idx, 0:-2]
				y_data = batch_xy_torch[idx:last_idx, -2:]
				loader.append([x_data, y_data])

		return loader


	if dataset is None:
		raise Exception('dataset not specified')
	
	if shuffle:
		random.shuffle(dataset)

	if n_data is not None and n_data < len(dataset):
		dataset = dataset[0:n_data]

	loader = batch_loader(env, dataset, device)

	return loader

def load_loader(name,batch_size,device):

	loader = []
	datadir = glob.glob("../preprocessed_data/batch_{}*.npy".format(name))
	for file in datadir: 
		
		batch_xy = np.load(file)
		# convert to torch
		batch_xy_torch = torch.from_numpy(batch_xy).float().to(device)

		# split data by batch size
		for idx in np.arange(0, batch_xy.shape[0], batch_size):
			last_idx = min(idx + batch_size, batch_xy.shape[0])
			# print("Batch of size ", last_idx - idx)
			x_data = batch_xy_torch[idx:last_idx, 0:-2]
			y_data = batch_xy_torch[idx:last_idx, -2:]
			loader.append([x_data, y_data])

	return loader


def train(param,env,model,optimizer,loader):
	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step
		prediction = model(b_x)     # input x and predict based on x
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		optimizer.zero_grad()   # clear gradients for next train
		loss.backward()         # backpropagation, compute gradients
		optimizer.step()        # apply gradients
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def test(param,env,model,loader):

	loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
	epoch_loss = 0
	for step, (b_x, b_y) in enumerate(loader): # for each training step
		prediction = model(b_x)     # input batch state and predict batch action
		loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
		epoch_loss += float(loss)
	return epoch_loss/(step+1)


def train_il(param, env, device):

	folder = os.path.dirname(param.il_train_model_fn)
	if not os.path.exists(folder):
		os.mkdir(folder)

	# seed(1) # numpy random gen seed 
	# torch.manual_seed(1)    # pytorch 

	print("Case: ",param.env_case)
	print("Controller: ",param.il_controller_class)

	# datasets
	if True:

		train_dataset = []
		test_dataset = [] 
		training = True 
		total_dataset_size = 0
	
		if not param.il_load_loader_on:

			shutil.rmtree('../preprocessed_data')
			os.mkdir('../preprocessed_data')

			for datapattern,num_data in param.datadict.items():
				print(os.getcwd())
				if param.env_name in ['SingleIntegrator']:
					datadir = glob.glob("../data/training/singleintegrator/central/*{}_ex*.npy".format(datapattern))
				elif param.env_name in ['DoubleIntegrator']:
					datadir = glob.glob("../data/training/doubleintegrator/central/*{}_ex*.npy".format(datapattern))

				len_case = 0
				with concurrent.futures.ProcessPoolExecutor() as executor:
					for dataset in executor.map(env.load_dataset_action_loss, datadir):
						if np.random.uniform(0, 1) <= param.il_test_train_ratio:
							train_dataset.extend(dataset)
						else:
							test_dataset.extend(dataset)

						len_case += len(dataset)
						print('num_agents,len_case = {},{}'.format(datapattern,len_case))

						if len_case > num_data:
							break

			print('Total Training Dataset Size: ',len(train_dataset))
			print('Total Testing Dataset Size: ',len(test_dataset))

			loader_train = make_loader(
				env,
				dataset=train_dataset,
				shuffle=True,
				batch_size=param.il_batch_size,
				n_data=param.il_n_data,
				name = "train",
				device=device)

			loader_test = make_loader(
				env,
				dataset=test_dataset,
				shuffle=True,
				batch_size=param.il_batch_size,
				n_data=param.il_n_data,
				name = "test",
				device=device)

		else:
			loader_train = load_loader("train",param.il_batch_size,device=device)
			# loader_train = load_loader("adaptive",param.il_batch_size)
			loader_test  = load_loader("test",param.il_batch_size,device=device)

	# init model
	if param.il_controller_class is 'Barrier':
		model = Barrier_Net(param,param.controller_learning_module).to(device)
	elif param.il_controller_class is 'Empty':
		model = Empty_Net(param,param.controller_learning_module).to(device)
	else:
		print('Error in Train Gains, programmatic controller not recognized')
		exit()

	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr, weight_decay = param.il_wd)
	adaptive_dataset_len_lst = []
	num_unique_points_lst = []
	if param.adaptive_dataset_on:

		# first, train on the whole dataset
		best_test_loss = Inf
		scheduler = ReduceLROnPlateau(optimizer, 'min')
		for epoch in range(1,param.il_n_epoch+1):
						
			train_epoch_loss = train(param,env,model,optimizer,loader_train)
			test_epoch_loss = test(param,env,model,loader_test)
			scheduler.step(test_epoch_loss)

			if epoch%param.il_log_interval==0:
				print('epoch: ', epoch)
				print('   Train Epoch Loss: ', train_epoch_loss)
				print('   Test Epoch Loss: ', test_epoch_loss)
				if test_epoch_loss < best_test_loss:
					best_test_loss = test_epoch_loss
					print('      saving @ best test loss:', best_test_loss)
					torch.save(model.to('cpu'), param.il_train_model_fn)
					model.to(device)

		index = Index()
		adaptive_dataset = list(train_dataset)
		# best_train_loss = Inf
		scheduler = ReduceLROnPlateau(optimizer, 'min')
		while len(adaptive_dataset)<param.ad_n_data:

			data = get_dynamic_dataset(model,env,param,index)
			adaptive_dataset.extend(data)
			adaptive_dataset_len_lst.append(len(adaptive_dataset)-len(train_dataset))
			num_unique_points_lst.append(index.get_total_stats())

			print('len(adaptive_dataset): ', len(adaptive_dataset)-len(train_dataset))
			print('total number of unique points: ', index.get_total_stats())
			
			loader_train = make_loader(
				env,
				dataset=adaptive_dataset,
				shuffle=True,
				batch_size=param.il_batch_size,
				n_data=param.il_n_data,
				name = "adaptive")

			# best_test_loss = Inf
			# scheduler = ReduceLROnPlateau(optimizer, 'min')
			for epoch in range(1,param.ad_n_epoch+1):
				
				train_epoch_loss = train(param,env,model,optimizer,loader_train)
				test_epoch_loss = test(param,env,model,loader_test)
				scheduler.step(train_epoch_loss)

				if epoch%param.il_log_interval==0:
					print('epoch: ', epoch)
					print('   Train Epoch Loss: ', train_epoch_loss)
					print('   Test Epoch Loss: ', test_epoch_loss)
				# 	if train_epoch_loss < best_train_loss:
						# best_test_loss = test_epoch_loss
						# print('      saving @ best test loss:', best_test_loss)
			torch.save(model,param.ad_train_model_fn)

		# index.print_stats()
		# np.save()

		# best_test_loss = Inf
		# scheduler = ReduceLROnPlateau(optimizer, 'min')
		# for epoch in range(1,param.il_n_epoch+1):
						
		# 	train_epoch_loss = train(param,env,model,optimizer,loader_train)
		# 	test_epoch_loss = test(param,env,model,loader_test)
		# 	scheduler.step(test_epoch_loss)

		# 	if epoch%param.il_log_interval==0:
		# 		print('epoch: ', epoch)
		# 		print('   Train Epoch Loss: ', train_epoch_loss)
		# 		print('   Test Epoch Loss: ', test_epoch_loss)
		# 		if test_epoch_loss < best_test_loss:
		# 			best_test_loss = test_epoch_loss
		# 			print('      saving @ best test loss:', best_test_loss)
		# 			torch.save(model,param.il_train_model_fn)

	else:

		with open(param.il_train_model_fn + ".csv", 'w') as log_file:
			log_file.write("time,epoch,train_loss,test_loss\n")
			start_time = time.time()
			best_test_loss = Inf
			scheduler = ReduceLROnPlateau(optimizer, 'min')
			for epoch in range(1,param.il_n_epoch+1):

				train_epoch_loss = train(param,env,model,optimizer,loader_train)
				test_epoch_loss = test(param,env,model,loader_test)
				scheduler.step(test_epoch_loss)

				if epoch%param.il_log_interval==0:
					print('epoch: ', epoch)
					print('   Train Epoch Loss: ', train_epoch_loss)
					print('   Test Epoch Loss: ', test_epoch_loss)
					if test_epoch_loss < best_test_loss:
						best_test_loss = test_epoch_loss
						print('      saving @ best test loss:', best_test_loss)
						torch.save(model.to('cpu'), param.il_train_model_fn)
						model.to(device)
				log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))

		# # debug loading memory usage
		# snapshot = tracemalloc.take_snapshot()
		# top_stats = snapshot.statistics('lineno')

		# print("[ Top 10 ]")
		# for stat in top_stats[:10]:
		# 	print(stat)

	# del model
	# torch.cuda.empty_cache()
	# print(torch.cuda.memory_stats())



