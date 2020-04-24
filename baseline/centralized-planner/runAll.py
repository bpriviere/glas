#!/usr/bin/env python3
import glob
import os
import subprocess

if __name__ == "__main__":

  # ./run.sh examples/swap2.yaml data/swap2.npy
  # for file in glob.glob("examples/empty-8-8-random-*_30_agents.yaml"):
  # for file in glob.glob("examples/*.yaml"):
#  for file in glob.glob("examples/map_8by8_obst12_agents10_ex*.yaml"):
  for file in glob.glob("examples/test_real_agent_locations.yaml"):
    basename = os.path.splitext(os.path.basename(file))[0]
    print(file)
    if not os.path.exists("data/{}.npy".format(basename)):
      subprocess.run("./run.sh {} data/{}.npy".format(file, basename), shell=True)
    else:
      print("Output file exists. Skipping.")
