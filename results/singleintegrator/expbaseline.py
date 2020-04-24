# execute from results folder!

import sys
import os
import argparse
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from train_il import train_il
from other_policy import Empty_Net_wAPF, GoToGoalPolicy
from sim import run_sim
import torch
import concurrent.futures
from itertools import repeat
import glob
from multiprocessing import cpu_count
from torch.multiprocessing import Pool
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.getcwd(),'singleintegrator'))
from createPlots import add_line_plot_agg, add_bar_agg, add_scatter
import stats
from matplotlib.backends.backend_pdf import PdfPages


def run_orca_r(file, r):
	basename = os.path.splitext(os.path.basename(file))[0]

	with tempfile.TemporaryDirectory() as tmpdirname:
		output_file = tmpdirname + "/orca.csv"
		subprocess.run("../baseline/orca/build/orca -i {} -o {} --Rsense {}".format(file, output_file, r), shell=True)
		# load file and convert to binary
		data = np.loadtxt(output_file, delimiter=',', skiprows=1, dtype=np.float32)
		# store in binary format
		folder = "singleintegrator/orcaR{}".format(r)
		if not os.path.exists(folder):
			os.mkdir(folder)
		with open("{}/{}.npy".format(folder, basename), "wb") as f:
			np.save(f, data, allow_pickle=False)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--orca', action='store_true')
	parser.add_argument('--apf', action='store_true')
	args = parser.parse_args()

	torch.multiprocessing.set_start_method('spawn')

	agents_lst = [2,4,8,16,32,64]
	obst_lst = [6,9,12]
	radii = [1,2,3,4,5,6,7,8] #[1,2,3,4]

	if args.apf:

		folder = "singleintegrator/apf"
		if not os.path.exists(folder):
			os.mkdir(folder)

		param = run_singleintegrator.SingleIntegratorParam()
		env = SingleIntegrator(param)

		controller = {
			'apf': Empty_Net_wAPF(param,env,GoToGoalPolicy(param,env))}

		for agent in agents_lst:
			for obst in obst_lst:
				files = glob.glob("singleintegrator/instances/*obst{}_agents{}_*.yaml".format(obst,agent), recursive=True)
				with Pool(12) as p:
					p.starmap(run_singleintegrator.run_batch, zip(repeat(param), repeat(env), files, repeat(controller)))

	if args.orca:
		for r in radii:
			for agent in agents_lst:
				for obst in obst_lst:
					files = glob.glob("singleintegrator/instances/*obst{}_agents{}_*.yaml".format(obst,agent), recursive=True)
					with Pool(12) as p:
						p.starmap(run_orca_r, zip(files, repeat(r)))
