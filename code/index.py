import glob
import numpy as np
import hnswlib
import os
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4

state_dim_per_agent = 2

class Index:

	def __init__(self):

		self.P = dict()
		self.D = dict()
		self.stats = dict()
		self.fileidx = 0


		datadir = glob.glob("../preprocessed_data/batch_train*.npy")
		for file in datadir:
			data = np.load(file)
			index_fn = "{}.index".format(file)
			num_neighbors = int(data[0,0])
			num_obstacles = int((data.shape[1] - 1 - state_dim_per_agent - num_neighbors*state_dim_per_agent - 2) / 2)
			print(file, num_neighbors, num_obstacles)

			# ignore first column (num_neighbors) and last two columns (actions)
			dim = data.shape[1] - 3 
			p = hnswlib.Index(space='l2', dim=dim)
			if os.path.exists(index_fn):
				p.load_index(index_fn)
			else:
				# see https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for params
				p.init_index(max_elements = data.shape[0],ef_construction=100, M=16)
				p.add_items(data[:,1:1+dim])
				p.save_index("{}.index".format(file))

			# p.set_num_threads(1)

			self.P[(num_neighbors,num_obstacles)] = p 
			self.D[(num_neighbors,num_obstacles)] = data
			self.stats[(num_neighbors,num_obstacles)] = np.zeros(data.shape[0])

	def query(self,obs,k):
		num_neighbors = int(obs[0,0])
		num_obstacles = int((obs.shape[1] - 1 - state_dim_per_agent - num_neighbors*state_dim_per_agent) / 2)

		if (num_neighbors,num_obstacles) not in self.P:
			return []

		p = self.P[(num_neighbors,num_obstacles)]
		stats = self.stats[(num_neighbors,num_obstacles)]
		labels, distances = p.knn_query(obs[:,1:], k=min(k, stats.shape[0]) )
		for l in labels[0]:
			stats[l] += 1
		# return [self.D[(num_neighbors,num_obstacles)][l] for l in labels[0]]

		# if (distances[0][0] > 2 or distances[0][-1] > 4) and self.fileidx < 100:
		if False:
			pp = PdfPages("index_query_{}.pdf".format(self.fileidx))
			self.plot_obs_doubleintegrator(pp, obs[0], "input", has_action = False)
			for k, l in enumerate(labels[0]):
				self.plot_obs_doubleintegrator(pp, self.D[(num_neighbors,num_obstacles)][l], "k = {}, dist = {}".format(k, distances[0][k]))
			pp.close()
			self.fileidx += 1

		return [(num_neighbors,num_obstacles,l) for l in labels[0]]

	def query_lst(self,obs_lst,k):
		results_set = set()
		for obs in obs_lst:
			results_set.update(self.query(obs,k))
		results_lst = []
		for nn,no,l in results_set:
			results_lst.append(self.D[(nn,no)][l])
		return results_lst

	def print_stats(self):
		total = 0
		for key, stat in self.stats.items():
			print("stats for ", key)
			print("  total ", np.sum(stat))
			print("  hit ", np.count_nonzero(stat) / stat.shape[0] * 100, " %")
			print("  max ", np.max(stat) / np.sum(stat) * 100, " %")
			total += np.count_nonzero(stat)
		print('total: ', total)

	def get_total_stats(self):
		total = 0
		for key, stat in self.stats.items():
			total += np.count_nonzero(stat)
		return total 


	def plot_obs_singleintegrator(self, pp, observation,title=None, has_action = True):
		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_autoscalex_on(False)
		ax.set_autoscaley_on(False)
		ax.set_title(title)

		# print(observation)
		num_neighbors = int(observation[0])
		if has_action:
			num_obstacles = int((observation.shape[0]-5 - 2*num_neighbors)/2)
		else:
			num_obstacles = int((observation.shape[0]-3 - 2*num_neighbors)/2)

		# print(observation, num_neighbors, num_obstacles)

		robot_pos = np.array([0,0])
		ax.add_patch(Circle(robot_pos, 0.2, facecolor='b', alpha=0.5))
		
		idx = 3
		for i in range(num_neighbors):
			pos = observation[idx : idx+2] + robot_pos
			ax.add_patch(Circle(pos, 0.2, facecolor='gray', edgecolor='red', alpha=0.5))
			idx += 2

		for i in range(num_obstacles):
			pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
			ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
			# pos = observation[idx : idx+2] + robot_pos
			# ax.add_patch(Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
			idx += 2

		# plot goal
		goal = observation[1:3] + robot_pos
		ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color='blue'))

		# plot action
		if has_action:
			plt.arrow(0,0,observation[-2],observation[-1])

		ax.add_patch(Circle(robot_pos, 3.0, facecolor='gray', edgecolor='black', alpha=0.1))

		# plt.show()
		pp.savefig(fig)
		plt.close(fig)

	def plot_obs_doubleintegrator(self, pp, observation,title=None, has_action = True):
		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_autoscalex_on(False)
		ax.set_autoscaley_on(False)
		ax.set_title(title)

		# print(observation)
		num_neighbors = int(observation[0])
		if has_action:
			num_obstacles = int((observation.shape[0]-7 - 4*num_neighbors)/2)
		else:
			num_obstacles = int((observation.shape[0]-5 - 4*num_neighbors)/2)

		# print(observation, num_neighbors, num_obstacles)

		robot_pos = np.array([0,0])
		ax.add_patch(Circle(robot_pos, 0.2, facecolor='b', alpha=0.5))
		
		X = []
		Y = []
		U = []
		V = []

		idx = 5
		for i in range(num_neighbors):
			pos = observation[idx : idx+2] + robot_pos
			X.append(pos[0])
			Y.append(pos[1])
			U.append(observation[idx+2])
			V.append(observation[idx+3])
			ax.add_patch(Circle(pos, 0.2, facecolor='gray', edgecolor='red', alpha=0.5))
			idx += 4

		for i in range(num_obstacles):
			pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
			ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
			# pos = observation[idx : idx+2] + robot_pos
			# ax.add_patch(Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
			idx += 2

		# plot goal
		goal = observation[1:3] + robot_pos
		ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color='blue'))
		X.append(robot_pos[0])
		Y.append(robot_pos[1])
		U.append(observation[3])
		V.append(observation[4])
		print(observation)

		# plot velocity vectors
		ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=1.0,color='red',width=0.005)

		# plot action
		if has_action:
			plt.arrow(0,0,observation[-2],observation[-1])

		ax.add_patch(Circle(robot_pos, 3.0, facecolor='gray', edgecolor='black', alpha=0.1))

		# plt.show()
		pp.savefig(fig)
		plt.close(fig)
