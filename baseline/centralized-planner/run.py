import argparse
import tempfile
import os
import subprocess
import yaml
import json

from discretePreProcessing import discretePreProcessing
from discretePostProcessing import discretePostProcessing
from exportTrajectories import exportTrajectories

RADIUS = 0.2

def createAddVertexAndAgentFiles(input_fn, addVertex_fn, agents_fn):
	with open(input_fn) as file:
		instance = yaml.load(file, Loader=yaml.SafeLoader)

	vertices = []
	agents = []
	for i, agent in enumerate(instance["agents"]):
		start = [agent["start"][0] + 0.5, agent["start"][1] + 0.5, 0.23]
		goal = [agent["goal"][0] + 0.5, agent["goal"][1] + 0.5, 0.23]
		print(start)
		vertices.append({'name': 'start' + str(i), 'pos': start})
		vertices.append({'name': 'goal' + str(i), 'pos': goal})
		agents.append({
			'name': agent["name"],
			'type': 'ground',
			'start': 'start' + str(i) + "_ground",
			'goal': 'goal' + str(i) + "_ground"})

	with open(addVertex_fn, "w") as f:
		yaml.dump({'vertices': vertices}, f)

	with open(agents_fn, "w") as f:
		yaml.dump({'agents': agents}, f)


def convertSchedule(json_fn, yaml_fn):
	with open(json_fn) as file:
		data = json.load(file)

	agents = dict()
	for agent in data["agents"]:
		path = []
		for p in agent["path"]:
			path.append({
				't': float(p["t"]),
				'x': float(p['x']) - 0.5,
				'y': float(p['y']) - 0.5})
		agents[agent['name']] = path

	with open(yaml_fn, "w") as f:
		yaml.dump({'schedule': agents}, f)


def run(input_fn, output_fn, use_grid_planner = True):
	with tempfile.TemporaryDirectory() as tmpdirname:
		print('created temporary directory', tmpdirname)

		# discrete planning

		if use_grid_planner:
			discretePreProcessing(input_fn, os.path.join(tmpdirname, "input.yaml"))

			subprocess.run(["./multi-robot-trajectory-planning/build/libMultiRobotPlanning/ecbs",
				"-i", os.path.join(tmpdirname, "input.yaml"),
				"-o", os.path.join(tmpdirname, "discreteSchedule.yaml"),
				"-w", "1.3"], timeout=60)

			# postprocess output (split paths)
			discretePostProcessing(
				os.path.join(tmpdirname, "input.yaml"),
				os.path.join(tmpdirname, "discreteSchedule.yaml"),
				os.path.join(tmpdirname, "discreteSchedule.yaml"))

			# convert yaml map -> octomap
			subprocess.run(["./multi-robot-trajectory-planning/build/tools/map2octomap/map2octomap",
				"-m", input_fn,
				"-o", os.path.join(tmpdirname, "map.bt")])

			# convert octomap -> STL (visualization)
			# (skip, since we do batch processing here)

		else:
			# generate add Vertex file
			createAddVertexAndAgentFiles(input_fn,
				os.path.join(tmpdirname, "addVertices.yaml"),
				os.path.join(tmpdirname, "agents.yaml"))

			# convert yaml map -> octomap
			subprocess.run(["./multi-robot-trajectory-planning/build/tools/map2octomap/map2octomap",
				"-m", input_fn,
				"-o", os.path.join(tmpdirname, "map.bt")])

			# create roadmap
			subprocess.run(["./discrete/build/generateRoadmap/generateRoadmap",
				"-e", os.path.join(tmpdirname, "map.bt"),
				"-r", "<Cylinder:{},0.45>".format(RADIUS),
				"-o", os.path.join(tmpdirname, "roadmap.yaml"),
				"-a", os.path.join(tmpdirname, "addVertices.yaml"),
				"-c", "config/roadmapConfigTypeGround.yaml",
				# "--type", "SPARS",
				"--type", "Grid",
				"--dimension", "2",
				"--fixedZ", "0.23"])

			# annotate roadmap
			subprocess.run(["./discrete/build/annotateRoadmap/annotateRoadmap",
				"-t", "config/types.yaml",
				"--folder", tmpdirname,
				"-o", os.path.join(tmpdirname, "annotatedRoadmap.yaml")])

			# run ecbs
			subprocess.run(["./discrete/build/ecbsCLI/ecbsCLI",
				"-m", os.path.join(tmpdirname, "annotatedRoadmap.yaml"),
				"-a", os.path.join(tmpdirname, "agents.yaml"),
				"-f", "3.0",
				"-o", os.path.join(tmpdirname, "discreteSchedule.json")], timeout=120)

			convertSchedule(
				os.path.join(tmpdirname, "discreteSchedule.json"),
				os.path.join(tmpdirname, "discreteSchedule.yaml"))


		# continuous planning
		cmd = "path_setup,smoothener_batch('{}','{}','{}','{}'),quit".format(
			os.path.join(tmpdirname, "map.bt"),
			os.path.join(tmpdirname, "discreteSchedule.yaml"),
			"../../config/types.yaml",
			os.path.join(tmpdirname) + "/")
		subprocess.run(["matlab",
			"-nosplash",
			"-nodesktop",
			"-r", cmd],
			cwd="multi-robot-trajectory-planning/smoothener",
			timeout=5*60)

		exportTrajectories(
			tmpdirname,
			"config/types.yaml",
			input_fn,
			output_fn)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="input file (yaml)")
	parser.add_argument("output", help="output file (npy)")
	args = parser.parse_args()

	run(args.input, args.output, False)