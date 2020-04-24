import glob
import os
import numpy as np
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle

import nnexport

plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

# some parameters
r_comm = 3 
max_neighbors = 5
max_obstacles = 5 
dx = 0.25


def plot_policy_vector_field(fig,ax,map_data,i):
	
	obstacles = map_data["map"]["obstacles"]
	transformation = [[np.eye(2)]]

	X = np.arange(0,map_data["map"]["dimensions"][0]+dx,dx)
	Y = np.arange(0,map_data["map"]["dimensions"][1]+dx,dx)
	U = np.zeros((len(X),len(Y)))
	V = np.zeros((len(X),len(Y)))
	C = np.zeros((len(X),len(Y)))

	o = [] 
	for i_x,x in enumerate(X):
		for i_y,y in enumerate(Y):
			if not collision((x,y),obstacles):

				a = policy(map_data,x,y,i)

				U[i_y,i_x] = a[0]
				V[i_y,i_x] = a[1]
				C[i_y,i_x] = np.linalg.norm( np.array(a))

	ax.quiver(X,Y,U,V,C)


def collision(p,o_lst):
	return False


def policy(map_data,x,y,i):

	p = np.array([x,y])
	nn = len(map_data["agents"])
	no = len(map_data["map"]["obstacles"])

	goal = np.array(map_data["agents"][i]["goal"])
	relative_goal = goal - p + 0.5
	scale = r_comm/np.linalg.norm(relative_goal)
	if scale < 1:
		relative_goal = scale*relative_goal

	relative_neighbors = []
	for j in range(nn):
		if not j == i:
			s_j = map_data["agents"][j]["start"]
			if np.linalg.norm(s_j-p) < r_comm:
				relative_neighbors.append(s_j - p)
	if len(relative_neighbors)>max_neighbors:
		relative_neighbors = sorted(relative_neighbors, key=lambda x: np.linalg.norm(x))
		relative_neighbors = relative_neighbors[0:max_neighbors]

	relative_obstacles = []
	for o in map_data["map"]["obstacles"]:
		p_o = np.array(o,dtype=float) + 0.5
		relative_obstacles.append(p_o - p) 
	if len(relative_obstacles)>max_neighbors:
		relative_obstacles = sorted(relative_obstacles, key=lambda x: np.linalg.norm(x))
		relative_obstacles = relative_obstacles[0:max_obstacles]

	# apply policy
	nnexport.nn_reset()
	for n in relative_neighbors:
		nnexport.nn_add_neighbor(n)
	for o in relative_obstacles:
		nnexport.nn_add_obstacle(o)
	return nnexport.nn_eval(relative_goal)


if __name__ == '__main__':

	instance = "map_8by8_obst6_agents4_ex0002"

	# load map 
	# instance_fn = "../results/singleintegrator/instances/{}.yaml".format(instance)
	instance_fn = "flighttest.yaml"
	with open(instance_fn) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)
	print(map_data)
	num_agents = len(map_data["agents"])

	# which agent to show 
	i = 0

	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	ax.set_xlim((-1,9))
	ax.set_ylim((-1,9))

	plot_policy_vector_field(fig,ax,map_data,i)

	for o in map_data["map"]["obstacles"]:
		ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
	for x in range(-1,map_data["map"]["dimensions"][0]+1):
		ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
		ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
	for y in range(map_data["map"]["dimensions"][0]):
		ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
		ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

	color = 'black'
	for j in range(num_agents):
		if i == j:
			goal = np.array(map_data["agents"][j]["goal"])
			ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))
		else:
			start = np.array(map_data["agents"][j]["start"])
			ax.add_patch(Circle(start, 0.2, alpha=0.5, color=color))

	plt.show(fig)
