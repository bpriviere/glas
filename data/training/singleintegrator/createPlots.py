import glob
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


if __name__ == '__main__':

	pp = PdfPages("results_obst6_agents10.pdf")

	for file in sorted(glob.glob("central/*obst6_agents10*.npy")):
	# for file in sorted(glob.glob("central_single_case_2/*.npy")):
	# for file in sorted(glob.glob("central/*agents2_*")):
		instance = os.path.splitext(os.path.basename(file))[0]
		print(instance)
		map_filename = "instances/{}.yaml".format(instance)
		with open(map_filename) as map_file:
			map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

		fig, ax = plt.subplots()
		ax.set_title(instance, fontdict={'fontsize': 12})
		ax.set_aspect('equal')

		for o in map_data["map"]["obstacles"]:
			ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
		for x in range(-1,map_data["map"]["dimensions"][0]+1):
			ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
			ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
		for y in range(map_data["map"]["dimensions"][0]):
			ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
			ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

		data = np.load(file)
		num_agents = len(map_data["agents"])
		for i in range(num_agents):
			line = ax.plot(data[:,1+i*4], data[:,1+i*4+1])
			color = line[0].get_color()
			start = np.array(map_data["agents"][i]["start"])
			goal = np.array(map_data["agents"][i]["goal"])
			ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
			ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

		pp.savefig(fig)
		plt.close(fig)

	pp.close()
