#!/usr/bin/env python3
import glob
import os
import subprocess
import numpy as np
import argparse
import concurrent.futures
import tempfile

def rollout_instance(file):
  basename = os.path.splitext(os.path.basename(file))[0]
  print(basename)
  if args.central:
    # Centralized planner
    if not os.path.exists(os.path.abspath("central/"+basename+".npy")):
      subprocess.run("python3 run.py {} {}".format(
        os.path.abspath(file),
        os.path.abspath("central/"+basename+".npy")),
        shell=True,
        cwd="../../baseline/centralized-planner")

  if args.orca:
    # ORCA
    with tempfile.TemporaryDirectory() as tmpdirname:
      output_file = tmpdirname + "/orca.csv"
      subprocess.run("../../baseline/orca/build/orca -i {} -o {}".format(file, output_file), shell=True)
      # load file and convert to binary
      data = np.loadtxt(output_file, delimiter=',', skiprows=1, dtype=np.float32)
      # store in binary format
      if not os.path.exists("orca"):
        os.mkdir("orca")
      with open("orca/{}.npy".format(basename), "wb") as f:
          np.save(f, data, allow_pickle=False)

  if args.il:
    subprocess.run("python3 examples/run_singleintegrator.py -i {} --batch".format(os.path.abspath(file)),
      cwd="../../code",
      shell=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--orca", action='store_true')
  parser.add_argument("--central", action='store_true')
  parser.add_argument("--il", action='store_true')
  args = parser.parse_args()


  # datadir = sorted(glob.glob("instances/*"))

  agents_lst = [64] #[2,4,8,16,32,64]
  obst_lst = [6] #int(map_size[0] * map_size[1] * 0.1)

  datadir = []
  for agents in agents_lst:
    for obst in obst_lst:
      datadir.extend(glob.glob("instances/*obst{}_agents{}_*".format(obst,agents)))

  datadir = sorted(datadir)

  datadir = datadir[0:2]

  # Serial version
  # for file in datadir:
  #   rollout_instance(file)

  # parallel version
  with concurrent.futures.ThreadPoolExecutor() as executor:
    for _ in executor.map(rollout_instance, datadir):
      pass
