# hack to be able to load modules in Python3
import sys, os
sys.path.insert(1, os.path.join(os.getcwd(),'.'))

import argparse
import torch

from train_il import train_il
from sim import sim


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--il", action='store_true', help="Run Imitation Learning")
	parser.add_argument("--animate", action='store_true')
	parser.add_argument("-i", "--instance", help="File instance to run simulation on")
	parser.add_argument("--batch", action='store_true', help="use batch (npy) output instead of interactive (pdf) output")
	parser.add_argument("--export", action='store_true', help="export IL model to onnx")
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	args = parser.parse_args()
	return args


def run(param, env, controllers, initial_state = None, args = None):
	if args is None:
		args = parse_args()

	if not args.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if args.il:
		train_il(param, env, device)
	else:
		sim(param, env, controllers, initial_state, args.animate)
