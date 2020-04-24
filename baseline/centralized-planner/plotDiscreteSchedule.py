import numpy as np
import matplotlib.pyplot as plt
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import argparse
import yaml
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("schedule")
	args = parser.parse_args()

	with open(args.map) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	with open(args.schedule) as f:
		schedule_data = yaml.load(f, Loader=yaml.SafeLoader)

	fig, ax0 = plt.subplots(1,1)
	ax0.set_aspect('equal', adjustable='box')
	for o in map_data["map"]["obstacles"]:
		ax0.add_patch(plt.Rectangle(o, 1.0, 1.0))
	ax0.set_xlim(0, map_data["map"]["dimensions"][0])
	ax0.set_xlabel("x [m]")
	ax0.set_ylim(0, map_data["map"]["dimensions"][1])
	ax0.set_label("y [m]")
	ax0.set_title("X/Y Plot")

	for agent in schedule_data["schedule"]:
		print(agent)
		xs = []
		ys = []
		for p in schedule_data["schedule"][agent]:
			print(p)
			xs.append(p['x'])
			ys.append(p['y'])
		ax0.plot(xs, ys)

	plt.show()