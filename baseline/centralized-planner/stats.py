import numpy as np
import argparse
import yaml

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("schedule")
	args = parser.parse_args()

	data = np.load(args.schedule)

	with open(args.map) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	# find goal times
	goal_times = []
	for i, agent in enumerate(map_data["agents"]):
		goal = np.array([0.5,0.5]) + np.array(agent["goal"])
		distances = np.linalg.norm(data[:,(i*4+1):(i*4+3)] - goal, axis=1)
		lastIdx = np.max(np.argwhere(distances > 0.05))
		if lastIdx < data.shape[0] - 1:
			goal_time = data[lastIdx,0]
		else:
			print("Warning: Agent {} did not reach its goal! Last Dist: {}".format(i, distances[-1]))
			goal_time = float('inf')
		goal_times.append(goal_time)
	goal_times = np.array(goal_times)

	# Sum of cost:
	soc = np.sum(goal_times)

	# makespan
	makespan = np.max(goal_times)

	# control effort (here: single integrator => velocity)
	control_effort = 0
	for i, agent in enumerate(map_data["agents"]):
		control_effort += np.sum(np.abs(data[:,i*4+3]))
		control_effort += np.sum(np.abs(data[:,i*4+4]))
	control_effort *= (data[1,0] - data[0,0])

	# Collisions
	num_agents = len(map_data["agents"])
	min_dist = float('inf')
	for i in range(num_agents):
		for j in range(i+1, num_agents):
			pos_i = data[:,(i*4+1):(i*4+3)]
			pos_j = data[:,(j*4+1):(j*4+3)]
			distances = np.linalg.norm(pos_i - pos_j, axis=1)
			min_dist = min(min_dist, np.min(distances))
	print("Min dist: ", min_dist)



	print("SOC ", soc)
	print("Makespan ", makespan)
	print("Total control effort ", control_effort)
