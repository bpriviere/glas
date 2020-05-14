
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
from multiprocessing import cpu_count
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

def make_loader(
	env,
	param,
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

			with open("../{}/batch_{}_nn{}_no{}.npy".format(param.preprocessed_data_dir,name,num_neighbors,num_obstacles), "wb") as f:
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

def load_loader(name,batch_size,device,param):

	loader = []
	datadir = glob.glob("../{}/batch_{}*.npy".format(param.preprocessed_data_dir,name))
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
	train_dataset = []
	test_dataset = [] 
	training = True 
	total_dataset_size = 0

	if not param.il_load_loader_on:

		shutil.rmtree('../{}'.format(param.preprocessed_data_dir))
		os.mkdir('../{}'.format(param.preprocessed_data_dir))

		for datapattern,num_data in param.datadict.items():
			if param.env_name in ['SingleIntegrator']:
				datadir = glob.glob("../data/training/singleintegrator/central/*{}*.npy".format(datapattern))
			elif param.env_name in ['DoubleIntegrator']:
				datadir = glob.glob("../data/training/doubleintegrator/central/*{}*.npy".format(datapattern))
			random.shuffle(datadir)

			len_case = 0
			with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
				for dataset in executor.map(env.load_dataset_action_loss, datadir):
			# for file in datadir:
				# dataset = env.load_dataset_action_loss(file)
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
			param,
			dataset=train_dataset,
			shuffle=True,
			batch_size=param.il_batch_size,
			n_data=param.il_n_data,
			name = "train",
			device=device)

		loader_test = make_loader(
			env,
			param,
			dataset=test_dataset,
			shuffle=True,
			batch_size=param.il_batch_size,
			n_data=param.il_n_data,
			name = "test",
			device=device)

	else:
		loader_train = load_loader("train",param.il_batch_size,device,param)
		# loader_train = load_loader("adaptive",param.il_batch_size)
		loader_test  = load_loader("test",param.il_batch_size,device,param)

	# init model
	if param.il_controller_class is 'Barrier':
		model = Barrier_Net(param).to(device)
	elif param.il_controller_class is 'Empty':
		model = Empty_Net(param).to(device)
	else:
		print('Error in Train Gains, programmatic controller not recognized')
		exit()

	if param.il_pretrain_weights_fn is not None:
		model.load_weights(param.il_pretrain_weights_fn)

	optimizer = torch.optim.Adam(model.parameters(), lr=param.il_lr, weight_decay = param.il_wd)

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
					model.save_weights(param.il_train_model_fn + ".tar")
					model.to(device)
			log_file.write("{},{},{},{}\n".format(time.time() - start_time, epoch, train_epoch_loss, test_epoch_loss))