import numpy as np
import argparse
import yaml
import os

robot_radius = 0.15
goal_dist = 0.5 # minimum distance to count as goal

# from https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
def is_collision_circle_rectangle(circle_pos, circle_r, rect_tl, rect_br):
	# Find the closest point to the circle within the rectangle
	closest = np.clip(circle_pos, rect_tl, rect_br)
	# Calculate the distance between the circle's center and this closest point
	dist = np.linalg.norm(circle_pos - closest, axis=1)
	# If the distance is less than the circle's radius, an intersection occurs
	return dist + 1e-4 < circle_r, dist


def stats(map_filename, schedule_filename):

	data = np.load(schedule_filename)

	with open(map_filename) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	# find goal times
	goal_times = []
	num_agents_reached_goal = 0
	agents_reached_goal = set()
	for i, agent in enumerate(map_data["agents"]):
		goal = np.array([0.5,0.5]) + np.array(agent["goal"])
		distances = np.linalg.norm(data[:,(i*4+1):(i*4+3)] - goal, axis=1)
		goalIdx = np.argwhere(distances > goal_dist)
		if len(goalIdx) == 0:
			goalIdx = np.array([0])
		lastIdx = np.max(goalIdx)
		if lastIdx < data.shape[0] - 1:
			goal_time = data[lastIdx,0]
			num_agents_reached_goal += 1
			agents_reached_goal.add(i)
		else:
			# print("Warning: Agent {} did not reach its goal! Last Dist: {}".format(i, distances[-1]))
			goal_time = float('inf')
		goal_times.append(goal_time)
	goal_times = np.array(goal_times)

	# Sum of cost:
	soc = np.sum(goal_times)
	num_agents = len(map_data["agents"])

	# makespan
	makespan = np.max(goal_times)

	# control effort (here: single integrator => velocity)
	control_effort = np.zeros(num_agents)
	for i, agent in enumerate(map_data["agents"]):
		# print(np.linalg.norm(data[:,i*4+3:i*4+5], axis=1))
		control_effort[i] += np.sum(np.linalg.norm(data[:,i*4+3:i*4+5], axis=1))
		# np.sum(np.abs(data[:,i*4+3]))
		# control_effort[i] += np.sum(np.abs(data[:,i*4+4]))
	control_effort *= (data[1,0] - data[0,0])

	# Collisions
	agents_collided = set()
	# min_dist = float('inf')
	num_agent_agent_collisions = 0
	for i in range(num_agents):
		pos_i = data[:,(i*4+1):(i*4+3)]
		for j in range(i+1, num_agents):
			pos_j = data[:,(j*4+1):(j*4+3)]
			distances = np.linalg.norm(pos_i - pos_j, axis=1)
			inc = np.count_nonzero(distances < 2 * robot_radius - 1e-4)
			num_agent_agent_collisions += inc
			if inc > 0:
				print("a2a, ", np.min(distances))
				agents_collided.add(i)
				agents_collided.add(j)

	num_agent_obstacle_collisions = 0

	for i in range(num_agents):
		pos_i = data[:,(i*4+1):(i*4+3)]
		for o in map_data["map"]["obstacles"]:
			coll, dist = is_collision_circle_rectangle(pos_i, robot_radius, np.array(o), np.array(o) + np.array([1.0,1.0]))
			inc = np.count_nonzero(coll)

			# distances = np.linalg.norm(pos_i - (np.array(o) + np.array([0.5,0.5])), axis=1)
			# inc = np.count_nonzero(distances < 0.5+0.2 - 1e-4)
			num_agent_obstacle_collisions += inc
			if inc > 0:
				print("a2o, ", np.min(dist))
				agents_collided.add(i)

	num_agents_success = len(agents_reached_goal - agents_collided)

	result = dict()
	# result["min_dist"] = min_dist
	result["sum_time"] = soc
	result["makespan"] = makespan
	result["control_effort"] = control_effort
	result["control_effort_sum"] = np.sum(control_effort)
	result["control_effort_mean"] = np.mean(control_effort)
	result["num_agents_reached_goal"] = num_agents_reached_goal
	result["percent_agents_reached_goal"] = num_agents_reached_goal / num_agents * 100
	result["num_agent_agent_collisions"] = num_agent_agent_collisions
	result["num_agent_obstacle_collisions"] = num_agent_obstacle_collisions
	result["num_collisions"] = num_agent_agent_collisions + num_agent_obstacle_collisions
	result["num_agents_success"] = num_agents_success
	result["percent_agents_success"] = num_agents_success / num_agents * 100
	result["agents_succeeded"] = agents_reached_goal - agents_collided


	solver = os.path.dirname(schedule_filename)
	if '_' in solver:
		r = solver.split('_')
		solver = "_".join(r[0:-1])
		num_model = r[-1]
	else:
		num_model = 0

	result["num_agents"] = num_agents
	result["solver"] = solver
	result["num_model"] = num_model

	return result


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("schedule")
	args = parser.parse_args()

	print(stats(args.map, args.schedule))