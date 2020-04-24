#!/usr/bin/env python3

# execute from results folder!

import sys
import os
import argparse
sys.path.insert(1, os.path.join(os.getcwd(),'../code'))
sys.path.insert(1, os.path.join(os.getcwd(),'../code/examples'))
import run_singleintegrator
import run_singleintegrator_vel_sensing
from systems.singleintegrator import SingleIntegrator
# from systems.singleintegrator_vel_sensing import SingleIntegratorVelSensing
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

  agents_lst = [2,16]
  obst_lst = [6]
  # datasource = ["obst06_agents004", "obst06_agents008", "obst06_agents016",
                # "obst12_agents004", "obst12_agents008", "obst12_agents016","mixed"]
  datasource = ["obst06_agents004", "obst06_agents016",
                "mixed"]
  num_data = 250000

  if args.plot:
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.linewidth'] = 4

    solvers = {
      'orcaR3': 'ORCA',
      'central': 'Central',
      'apf': 'BF',
    }
    for src in datasource:
      # solvers['exp3BarrierS'+src] = src
      solvers['exp3EmptyS'+src] = src

    for obst in obst_lst:
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
        result["filename"] = file

        if instance in result_by_instance:
          result_by_instance[instance].append(result)
        else:
          result_by_instance[instance] = [result]

      # create plots
      pp = PdfPages("exp3_{}.pdf".format(obst))

      add_line_plot_agg(pp, result_by_instance, "percent_agents_success",
        x_label="number of robots",
        y_label="robot success [%]")
      add_line_plot_agg(pp, result_by_instance, "control_effort_mean",
        x_label="number of robots",
        y_label="average control effort of successful robots")
      add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

      # # TEMP TEMP
      # import yaml
      # from matplotlib.patches import Rectangle, Circle
      # for instance in sorted(result_by_instance):
      #   print(instance)
      #   results = result_by_instance[instance]

      #   # add_bar_chart(pp, results, "percent_agents_reached_goal", instance + " (% reached goal)")
      #   # add_bar_chart(pp, results, "num_collisions", instance + " (# collisions)")

      #   map_filename = "singleintegrator/instances/{}.yaml".format(instance)
      #   with open(map_filename) as map_file:
      #     map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

      #   for r in results:
      #     print("state space" + r["solver"])
      #     fig, ax = plt.subplots()
      #     ax.set_title("State Space " + r["solver"])
      #     ax.set_aspect('equal')

      #     for o in map_data["map"]["obstacles"]:
      #       ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
      #     for x in range(-1,map_data["map"]["dimensions"][0]+1):
      #       ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
      #       ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
      #     for y in range(map_data["map"]["dimensions"][0]):
      #       ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
      #       ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

      #     data = np.load(r["filename"])
      #     num_agents = len(map_data["agents"])
      #     dt = data[1,0] - data[0,0]
      #     for i in range(num_agents):
      #       # plot trajectory
      #       line = ax.plot(data[:,1+i*4], data[:,1+i*4+1],alpha=0.5)
      #       color = line[0].get_color()

      #       # plot velocity vectors:
      #       X = []
      #       Y = []
      #       U = []
      #       V = []
      #       for k in np.arange(0,data.shape[0], int(5.0 / dt)):
      #         X.append(data[k,1+i*4+0])
      #         Y.append(data[k,1+i*4+1])
      #         U.append(data[k,1+i*4+2])
      #         V.append(data[k,1+i*4+3])

      #       ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=0.5,color=color,width=0.005)

      #       # plot start and goal
      #       start = np.array(map_data["agents"][i]["start"])
      #       goal = np.array(map_data["agents"][i]["goal"])
      #       ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
      #       ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

      #     pp.savefig(fig)
      #     plt.close(fig)

      # # END TEMP TEMP


      pp.close()

    # plot loss curve
    pp = PdfPages("exp3_loss.pdf")
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

  for i in range(0,1):
    # train policy
    

    for src in datasource:
      if args.train:
        for cc in ['Empty']:
          param = run_singleintegrator.SingleIntegratorParam()
          # param = run_singleintegrator_vel_sensing.SingleIntegratorVelSensingParam()
          param.il_load_loader_on = False
          param.il_controller_class = cc
          param.datadict = dict()
          if src == "mixed":
            for src in datasource:
              param.datadict[src] = num_data / (len(datasource)-1)
          else:
            param.datadict[src] = num_data

          param.il_train_model_fn = 'singleintegrator/exp3{}S{}_{}/il_current.pt'.format(cc,src,i)
          env = SingleIntegrator(param)
          # env = SingleIntegratorVelSensing(param)
          train_il(param, env, device)

          del env
          del param

      elif args.sim:
        param = run_singleintegrator.SingleIntegratorParam()
        env = SingleIntegrator(param)
        # param = run_singleintegrator_vel_sensing.SingleIntegratorVelSensingParam()
        # env = SingleIntegratorVelSensing(param)
        
        # evaluate policy
        controllers = {
          # 'exp3BarrierS{}_{}'.format(src,i): Empty_Net_wAPF(param,env,torch.load('singleintegrator/exp3BarrierS{}_{}/il_current.pt'.format(src,i))),
          'exp3EmptyS{}_{}'.format(src,i): Empty_Net_wAPF(param,env,torch.load('singleintegrator/exp3EmptyS{}_{}/il_current.pt'.format(src,i))),
        }

        # for instance in instances:
          # run_singleintegrator.run_batch(instance, controllers)

        with Pool(24) as p:
          # p.starmap(run_singleintegrator.run_batch, zip(repeat(param), repeat(env), instances, repeat(controllers)))
          p.starmap(run_singleintegrator_vel_sensing.run_batch, zip(repeat(param), repeat(env), instances, repeat(controllers)))

        # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        #   for _ in executor.map(run_singleintegrator.run_batch, instances, repeat(controllers)):
        #     pass

