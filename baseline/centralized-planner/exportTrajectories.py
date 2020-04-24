#!/usr/bin/env
import yaml
import numpy as np
import argparse
import os

import uav_trajectory

# this script first finds the common stretchtime (identical to multi-robot-trajectory-planning/tools/scaleTrajectory.py)
# and then outputs a single csv file with the sampled result

# returns maximum velocity, acceleration
def findMaxDynamicLimits(traj):
  vmax = 0
  amax = 0
  for t in np.arange(0, traj.duration, 0.1):
    e = traj.eval(t)
    vmax = max(vmax, np.linalg.norm(e.vel))
    amax = max(amax, np.linalg.norm(e.acc))
  return vmax, amax

# returns average velocity, acceleration
def findAvgVel(traj):
  vel = []
  for t in np.arange(0.0 , traj.duration, 0.5):
    e = traj.eval(t)
    vel.append(np.linalg.norm(e.vel))
  print(traj.duration,np.mean(vel))
  return np.mean(vel)

# returns upper bound stretchtime factor
def upperBound(traj, vmax, amax):
  stretchtime = 1.0
  while True:
    v,a = findMaxDynamicLimits(traj)
    if v == 0 and a == 0:
      return stretchtime
    if v <= vmax and a <= amax:
      # print(v,a)
      return stretchtime
    traj.stretchtime(2.0)
    stretchtime = stretchtime * 2.0

# returns lower bound stretchtime factor
def lowerBound(traj, vmax, amax):
  stretchtime = 1.0
  while True:
    v,a = findMaxDynamicLimits(traj)
    if v == 0 and a == 0:
      return stretchtime
    if v >= vmax and a >= amax:
      return stretchtime
    traj.stretchtime(0.5)
    stretchtime = stretchtime * 0.5

def findStretchtime(file, vmax, amax):
  traj = uav_trajectory.Trajectory()
  traj.loadcsv(file)
  L = lowerBound(traj, vmax, amax)
  traj.loadcsv(file)
  U = upperBound(traj, vmax, amax)
  while True:
    # print("L ", L)
    # print("U ", U)
    if U - L < 0.1:
      return U
    middle = (L + U) / 2
    # print("try: ", middle)
    traj.loadcsv(file)
    traj.stretchtime(middle)
    v,a = findMaxDynamicLimits(traj)
    # print("v,a ", v, a)
    if v <= vmax and a <= amax:
      U = middle
    else:
      L = middle

# returns upper bound stretchtime factor
def upperBound2(traj, vavg):
  stretchtime = 1.0
  while True:
    v = findAvgVel(traj)
    if v == 0:
      return stretchtime
    if v <= vavg:
      # print(v,a)
      return stretchtime
    traj.stretchtime(2.0)
    stretchtime = stretchtime * 2.0

# returns lower bound stretchtime factor
def lowerBound2(traj, vavg):
  stretchtime = 1.0
  while True:
    v = findAvgVel(traj)
    if v == 0:
      return stretchtime
    if v >= vavg:
      return stretchtime
    traj.stretchtime(0.5)
    stretchtime = stretchtime * 0.5

def findStretchtime2(file, vavg):
  print(file)
  traj = uav_trajectory.Trajectory()
  traj.loadcsv(file)
  L = lowerBound2(traj, vavg)
  traj.loadcsv(file)
  U = upperBound2(traj, vavg)
  while True:
    print("L ", L)
    print("U ", U)
    if U - L < 0.1:
      return U
    middle = (L + U) / 2
    print("try: ", middle)
    traj.loadcsv(file)
    traj.stretchtime(middle)
    v = findAvgVel(traj)
    if v <= vavg:
      U = middle
    else:
      L = middle

def exportTrajectories(folder, typesFile, agentsFile, outputFile):

  with open(typesFile) as file:
    types = yaml.load(file, Loader=yaml.SafeLoader)

  with open(agentsFile) as file:
    agents = yaml.load(file, Loader=yaml.SafeLoader)

  agentTypes = dict()
  for agentType in types["agentTypes"]:
    agentTypes[agentType["type"]] = agentType

  result = 0.0
  for agent in agents["agents"]:
    name = agent["name"]
    if "type" in agent:
      agentType = agentTypes[agent["type"]]
    else:
      agentType = agentTypes["ground"]
    vmax = agentType["v_max"]
    amax = agentType["a_max"]
    stretchtime = findStretchtime(os.path.join(folder, name + ".csv"), vmax, amax)
    print(name, stretchtime)
    result = max(result, stretchtime)

  print("common stretchtime: {}".format(result))

  # export file
  # find total duration
  T = 0
  trajs = []
  for agent in agents["agents"]:
    name = agent["name"]
    traj = uav_trajectory.Trajectory()
    traj.loadcsv(os.path.join(folder, name + ".csv"))
    traj.stretchtime(result)
    T = max(T, traj.duration)
    trajs.append(traj)

  ts = np.arange(0, T, 0.01)
  data = np.empty((len(ts), 1+4*len(trajs)), dtype=np.float32)
  # write sampled data
  for k, t in enumerate(ts):
    data[k, 0] = t
    for a, traj in enumerate(trajs):
      e = traj.eval(t)
      data[k,1+4*a+0] = e.pos[0]
      data[k,1+4*a+1] = e.pos[1]
      data[k,1+4*a+2] = e.vel[0]
      data[k,1+4*a+3] = e.vel[1]

  # store in binary format
  with open(outputFile, "wb") as f:
    np.save(f, data, allow_pickle=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("folder", type=str, help="input folder containing csv files")
  parser.add_argument("typesFile", help="types file for agent types (yaml)")
  parser.add_argument("agentsFile", help="agents file with agents (yaml)")
  parser.add_argument("outputFile", help="output file (npy)")
  args = parser.parse_args()

  exportTrajectories(args.folder, args.typesFile, args.agentsFile, args.outputFile)
