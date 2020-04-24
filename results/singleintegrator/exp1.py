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
import matplotlib.pyplot as plt
import numpy as np

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

  agents_lst = [2,4,8,16,32]
  obst_lst = [6,12]

  if args.plot:
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['lines.linewidth'] = 4

    solvers = {
      'orcaR3': 'ORCA',
      'exp1Empty': 'NN+BF',
      'central': 'Central',
      'exp1Barrier': 'NNwBF'
    }

    # default fig size is [6.4, 4.8]
    fig, axs = plt.subplots(2, len(obst_lst), sharex='all', sharey='row', figsize = [6.4 * 1.4, 4.8 * 1.4])

    pp = PdfPages("exp1_collisions.pdf")
    for column, obst in enumerate(obst_lst):
      files = []
      result_by_instance = dict()
      for solver in solvers.keys():
        for agent in agents_lst:
          files.extend( glob.glob("singleintegrator/{}*/*obst{}_agents{}_*.npy".format(solver,obst,agent), recursive=True))
      for file in files:
        instance = os.path.splitext(os.path.basename(file))[0]
        map_filename = "singleintegrator/instances/{}.yaml".format(instance)
        result = stats.stats(map_filename, file)
        result["solver"] = solvers[os.path.basename(result["solver"])]

        if instance in result_by_instance:
          result_by_instance[instance].append(result)
        else:
          result_by_instance[instance] = [result]

      # create plots

      add_line_plot_agg(None, result_by_instance, "percent_agents_success",
        ax=axs[0, column])
      add_line_plot_agg(None, result_by_instance, "control_effort_mean",
        ax=axs[1, column])
      add_scatter(pp, result_by_instance, "num_collisions", "# collisions")
    
    pp.close()
    
    pp = PdfPages("exp1.pdf")

    for column in range(0, 2):
      axs[1,column].set_xlabel("number of robots")
    
    axs[0,0].set_ylabel("robot success [%]")
    axs[0,0].set_ylim([35,105])
    axs[1,0].set_ylabel("control effort")

    axs[0,0].set_title("10 % obstacles")
    axs[0,1].set_title("20 % obstacles")

    for column in range(0, 2):
      for row in range(0, 2):
        axs[row, column].grid(which='both')

    fig.tight_layout()
    axs[0,0].legend()
    pp.savefig(fig)
    plt.close(fig)
    pp.close()

    # plot loss curve
    pp = PdfPages("exp1_loss.pdf")
    fig,ax = plt.subplots()    
    ax.set_yscale('log')
    
    for solver in solvers.keys():
      files = glob.glob("singleintegrator/{}*/*.csv".format(solver), recursive=True)
      if len(files) == 0:
        continue


      train_loss =[]
      test_loss = []
      for file in files:
        data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float32)
        train_loss.append(data[:,2])
        test_loss.append(data[:,3])

      train_loss_std = np.std(train_loss,axis=0)
      train_loss_mean = np.mean(train_loss,axis=0)

      test_loss_std = np.std(test_loss,axis=0)
      test_loss_mean = np.mean(test_loss,axis=0)

      print(test_loss_std)

      # line1 = ax.plot(data[:,1], train_loss_mean, label="train loss",linewidth=1)[0]
      line2 = ax.plot(data[:,1], test_loss_mean, label=solvers[solver],linewidth=1)[0]

      # ax.fill_between(data[:,1],
      #   train_loss_mean-train_loss_std,
      #   train_loss_mean+train_loss_std,
      #   facecolor=line1.get_color(),
      #   linewidth=1e-3,
      #   alpha=0.5)
      ax.fill_between(data[:,1],
        test_loss_mean-test_loss_std,
        test_loss_mean+test_loss_std,
        facecolor=line2.get_color(),
        linewidth=1e-3,
        alpha=0.5)
      
    plt.legend()
    pp.savefig(fig)
    plt.close(fig)
    pp.close()

    exit()

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("singleintegrator/instances/*obst{}_agents{}_*".format(obst,agents)))
  instances = sorted(datadir)

  first_training = False


  for i in range(1):
      # train policy
      param = run_singleintegrator.SingleIntegratorParam()
      env = SingleIntegrator(param)
      if args.train:
        for cc in ['Empty']:#, 'Barrier']:
          param = run_singleintegrator.SingleIntegratorParam()
          param.il_controller_class = cc
          param.il_train_model_fn = 'singleintegrator/exp1{}_{}/il_current.pt'.format(cc,i)
          # only load the data once
          if first_training:
            param.il_load_loader_on = False
            first_training = False
          else:
            param.il_load_loader_on = True
          env = SingleIntegrator(param)
          train_il(param, env, device)

      elif args.sim:
        # evaluate policy
        controllers = {
          'exp1Empty_{}'.format(i): Empty_Net_wAPF(param,env,torch.load('singleintegrator/exp1Empty_{}/il_current.pt'.format(i))),
          'exp1Barrier_{}'.format(i) : torch.load('singleintegrator/exp1Barrier_{}/il_current.pt'.format(i))
        }

        # for instance in instances:
          # run_singleintegrator.run_batch(instance, controllers)

        with Pool(32) as p:
          p.starmap(run_singleintegrator.run_batch, zip(repeat(param), repeat(env), instances, repeat(controllers)))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        #   for _ in executor.map(run_singleintegrator.run_batch, instances, repeat(controllers)):
        #     pass

