import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse
import yaml

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("roadmap")
	args = parser.parse_args()

	with open(args.map) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	with open(args.roadmap) as f:
		roadmap = yaml.load(f, Loader=yaml.SafeLoader)

	vertices = dict()
	for v in roadmap["vertices"]:
		vertices[v["name"]] = v["pos"][0:2]

	lines = []
	for e in roadmap["edges"]:
		lines.append([vertices[e["from"]], vertices[e["to"]]])

	fig, ax0 = plt.subplots(1,1)
	ax0.set_aspect('equal', adjustable='box')
	for o in map_data["map"]["obstacles"]:
		ax0.add_patch(plt.Rectangle(o, 1.0, 1.0))
	ax0.set_xlim(0, map_data["map"]["dimensions"][0])
	ax0.set_xlabel("x [m]")
	ax0.set_ylim(0, map_data["map"]["dimensions"][1])
	ax0.set_label("y [m]")
	ax0.set_title("X/Y Plot")

	ax0.add_collection(LineCollection(lines))


	plt.show()