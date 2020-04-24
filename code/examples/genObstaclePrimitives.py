
"""
Script to create primitive obstacles instances
"""

import os
import collections
import random
import copy
import yaml
import glob 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Circle
from math import sin,cos,pi,sqrt
from random import random, randint

def make_obstacles(map_size,i_case):
	
	shift = int(map_size[0]/2)
	
	if i_case == 1:
	
		obstacles = [ \
			(0+shift,1+shift),
			(1+shift,0+shift),
			]

	elif i_case == 2:
		
		shift = map_size[0]/2
	
		obstacles = [ \
			(0+shift,1+shift),
			(1+shift,0+shift),
			(0+shift,-1+shift),
			]

	return obstacles


def make_agents(n_agents,obstacles,r_agent,map_size,i_case):

	agents = []
	def uniform_in_box(box):
		xmin,ymin,xmax,ymax = box[:]
		x = random()*(xmax-xmin) + xmin
		y = random()*(ymax-ymin) + ymin
		return [x,y]

	def check_collision(agent_loc,obstacles):
		x,y = agent_loc[:]
		collision = False
		for o in obstacles:
			if x + r_agent > o[0]-0.5 and x - r_agent < o[0]+0.5 and \
				y + r_agent > o[1]-0.5 and y - r_agent < o[1]+0.5:
				
				collision = True
				break

		return collision

	for i in range(n_agents):
		
		shift = map_size[0]/2

		if i_case == 1:

			# [xmin,ymin,xmax,ymax]
			start_box = [3.5,3.5,4.5,4.5]
			goal_box = [4.5,5.,7.,7.]

			start = uniform_in_box(start_box)
			while check_collision(start,obstacles):
				start = uniform_in_box(start_box)

			goal = uniform_in_box(goal_box)
			while check_collision(goal,obstacles):
				goal = uniform_in_box(goal_box)

		elif i_case == 2:

			possible_start = [[4,4]]
			possible_goal = [[6,2],[6,3],[6,4],[6,5],[6,6]]

			start = possible_start[randint(0,len(possible_start)-1)]
			goal = possible_goal[randint(0,len(possible_goal)-1)]

		agent = dict()
		agent["name"] = "agent" + str(i)
		agent["start"] = start
		agent["goal"] = goal

		agents.append(agent)
	return agents


def writeFile(obstacles, map_size, agents, file_name):
	
	data = dict()
	data["map"] = dict()
	data["map"]["dimensions"] = map_size
	data["map"]["obstacles"] = [list(o) for o in obstacles]
	data["agents"] = agents
	
	with open(file_name, "w") as f:
		yaml.dump(data, f, indent=4, default_flow_style=None)



if __name__ == "__main__":

	map_size = [8, 8]
	num_agents = 1
	r_agent = 0.2
	i_case = 1
	n_trial = 5
	vis_on = False
	write_on = True
	
	obstacles = make_obstacles(map_size,i_case)

	for i_trial in range(n_trial):

		agents = make_agents(num_agents, obstacles, r_agent, map_size,i_case)

		if write_on:
			fn = "obstacle_primitives_case_{:03}_ex{:04}.yaml".format(i_case,i_trial)
			print(fn)
			writeFile(obstacles, map_size, agents, fn)

		if vis_on:
			# vis

			print('agents: ', agents)
			print('obstacles: ', obstacles)

			fig,ax = plt.subplots()
			for o in obstacles:
				ax.add_patch(Rectangle((o[0]-0.5,o[1]-0.5), 1.0, 1.0, facecolor='gray', alpha=0.5))
			for agent in agents:
				ax.add_patch(Circle((agent["start"][0],agent["start"][1]), radius=r_agent, facecolor='gray', alpha=0.5))
				ax.add_patch(Circle((agent["goal"][0],agent["goal"][1]), radius=r_agent, facecolor='gray', alpha=0.5))
				# ax.scatter(agent["start"][0], agent["start"][1])
				# ax.scatter(agent["goal"][0], agent["goal"][1])

			ax.set_xlim([0,map_size[0]])
			ax.set_ylim([0,map_size[1]])
			ax.set_aspect('equal')
			ax.grid(True)
			plt.show()
	