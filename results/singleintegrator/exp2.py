#!/usr/bin/env python3

# execute from results folder!

import sys
import os
import argparse
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from train_il import train_il
from other_policy import Empty_Net_wAPF
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
  parser.add_argument('--train', action='store_true', help='run training')
  parser.add_argument('--sim', action='store_true', help='run validation inference')
  parser.add_argument('--plot', action='store_true', help='create plots')
  args = parser.parse_args()

  torch.multiprocessing.set_start_method('spawn')

  if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  agents_lst = [2,4,8] #[2,4,8,16,32]
  obst_lst = [6] #[6,9,12]
  radii = [1,2,3,4,5,6,7,8] #[1,2,3,4]
  training_data = [100000, 500000, 1000000]
  training_data = [10000, 100000, 1000000]

  if args.plot:
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 4

    for obst in obst_lst:
      result_by_instance = dict()
      for r in radii:
        for agent in agents_lst:
          # load Empty
          for td in training_data:
            files = glob.glob("singleintegrator/exp2EmptyR{}td{}_*/*obst{}_agents{}_*.npy".format(r,td,obst,agent), recursive=True)
       
            for file in files:
              instance = os.path.splitext(os.path.basename(file))[0]
              map_filename = "singleintegrator/instances/{}.yaml".format(instance)
              result = stats.stats(map_filename, file)
              if td == 100000:
                result["solver"] = "NN+BF (100k training data)"
              elif td == 500000:
                result["solver"] = "NN+BF (500k training data)"
              elif td == 1000000:
                result["solver"] = "NN+BF (1M training data)"
              else:
                result["solver"] = "Empty{}".format(td)
              result["Rsense"] = r

              if instance in result_by_instance:
                result_by_instance[instance].append(result)
              else:
                result_by_instance[instance] = [result]

          # load ORCA
          files = glob.glob("singleintegrator/orcaR{}*/*obst{}_agents{}_*.npy".format(r,obst,agent), recursive=True)
     
          for file in files:
            instance = os.path.splitext(os.path.basename(file))[0]
            map_filename = "singleintegrator/instances/{}.yaml".format(instance)
            result = stats.stats(map_filename, file)
            result["solver"] = "ORCA"
            result["Rsense"] = r

            if instance in result_by_instance:
              result_by_instance[instance].append(result)
            else:
              result_by_instance[instance] = [result]

      # create plots
      pp = PdfPages("exp2_{}.pdf".format(obst))


      # add_bar_agg(pp, result_by_instance, "num_agents_success", "# robots success")
      add_line_plot_agg(pp, result_by_instance, "percent_agents_success", group_by="Rsense",
        x_label="sensing radius [m]",
        y_label="robot success [%]")
      # add_line_plot_agg(pp, result_by_instance, "control_effort_sum", "control effort")
      # add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

      pp.close()
    exit()

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("singleintegrator/instances/*obst{}_agents{}_*".format(obst,agents)))
  instances = sorted(datadir)


  for i in range(0,10):
    # train policy
    for r in radii:
      for td in training_data:
        if args.train:
          for cc in ['Empty']: #'Barrier']:
            param = run_singleintegrator.SingleIntegratorParam()
            param.r_comm = r
            param.r_obs_sense = r
            param.max_neighbors = 10000 #5
            param.max_obstacles = 10000 #5
            param.il_load_loader_on = False
            param.il_controller_class = cc
            param.datadict["4"] = td

            param.il_train_model_fn = 'singleintegrator/exp2{}R{}td{}_{}/il_current.pt'.format(cc,r,td,i)
            env = SingleIntegrator(param)
            train_il(param, env, device)
            del env
            del param

        elif args.sim:
          param = run_singleintegrator.SingleIntegratorParam()
          param.r_comm = r
          param.r_obs_sense = r
          param.max_neighbors = 10000 #5
          param.max_obstacles = 10000 #5

          env = SingleIntegrator(param)
          # evaluate policy
          controllers = {
            'exp2EmptyR{}td{}_{}'.format(r,td,i): Empty_Net_wAPF(param,env,torch.load('singleintegrator/exp2EmptyR{}td{}_{}/il_current.pt'.format(r,td,i))),
            'exp1BarrierR{}_td{}_{}'.format(r,td,i) : torch.load('singleintegrator/exp1Barrier_{}/il_current.pt'.format(r,td,i))
          }

          # for instance in instances:
            # run_singleintegrator.run_batch(instance, controllers)


          with Pool(12) as p:
            p.starmap(run_singleintegrator.run_batch, zip(repeat(param), repeat(env), instances, repeat(controllers)))

          # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
          #   for _ in executor.map(run_singleintegrator.run_batch, instances, repeat(controllers)):
          #     pass

