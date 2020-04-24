#!/usr/bin/env python3
import glob
import os
import subprocess
import numpy as np

if __name__ == "__main__":

  # ./run.sh examples/swap2.yaml data/swap2.npy
  # for file in glob.glob("examples/empty-8-8-random-1_30_agents.yaml"):
  # for file in glob.glob("examples/*.yaml"):
  for file in glob.glob("examples/map_8by8_obst12_agents5_ex0.yaml"):
    basename = os.path.splitext(os.path.basename(file))[0]
    print(file)
    subprocess.run("../orca/build/orca -i {}".format(file), shell=True)
    # load file and convert to binary
    data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
    # store in binary format
    with open("data-orca/{}.npy".format(basename), "wb") as f:
        np.save(f, data, allow_pickle=False)
