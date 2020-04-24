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
    if not os.path.exists(os.path.abspath("central3/"+basename+".npy")):
      subprocess.run("python3 run.py {} {}".format(
        os.path.abspath(file),
        os.path.abspath("central3/"+basename+".npy")),
        shell=True,
        cwd="../../baseline/centralized-planner")

  if args.orca:
    # ORCA
    with tempfile.TemporaryDirectory() as tmpdirname:
      output_file = tmpdirname + "/orca.csv"
      subprocess.run("../../baseline/orca/build/orca -i {} -o {} --robotRadius 0.21".format(file, output_file), shell=True)
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

  # datadir = glob.glob("instances/*obst06_agents032*.yaml")

  import random
  datadir = glob.glob("instances/*.yaml")
  random.shuffle(datadir)

  # files = glob.glob("instances/*obst6_agents4*.yaml")
  # datadir.extend(random.sample(files, min(len(files), 10000)))

  # files = glob.glob("instances/*obst06_agents032*.yaml")
  # datadir.extend(random.sample(files, min(len(files), 10000)))

  # # Serial version
  # for file in datadir:
  #   rollout_instance(file)

  # parallel version
  with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
    for _ in executor.map(rollout_instance, datadir):
      pass
