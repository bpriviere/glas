import numpy as np
# import scipy
# import scipy.spatial
# import nmslib
import hnswlib
import rowan

import matplotlib.pyplot as plt

def rrt(param, env):

	batch_size = 250
	num_actions_per_step = 4
	check_goal_iter = 1000
	sample_goal_iter = 10
	eps = 0.1
	num_actions_per_steer = 1

	# initialize with start state
	x0 = env.reset()
	data = np.zeros((50000, env.n + env.m + 1))
	data[0,0:env.n] = x0
	# data[0,5] = 0
	data[0,env.n+env.m] = -1

	# goal state
	xf = param.ref_trajectory[:,-1]

	print("Plan from: {} to {}".format(x0, xf))

	# index = nmslib.init(method='hnsw', space='l2')
	# index.addDataPoint(0, data[0])
	# index.createIndex(print_progress=False)

	index = hnswlib.Index(space='l2', dim=env.n)
	# ef_construction - controls index search speed/build speed tradeoff
	#
	# M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
	# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
	index.init_index(max_elements=data.shape[0], ef_construction=100, M=16)

	# Controlling the recall by setting ef:
	# higher ef leads to better accuracy, but slower search
	index.set_ef(10)

	index.add_items(data[0:1,0:env.n])

	i = 1
	last_index_update_idx = 1
	no_solution_count = 0
	while i < data.shape[0] and no_solution_count < 500:
		# randomly sample a state
		if i % sample_goal_iter == 0:
			x = xf
		else:
			#x = env.reset()
			x = np.random.uniform(env.s_min, env.s_max)

		# find closest state in tree
		pdist = np.linalg.norm(data[0:i,0:3] - x[0:3], axis=1)
		vdist = np.linalg.norm(data[0:i,3:6] - x[3:6], axis=1)
		qdist = rowan.geometry.sym_distance(data[0:i,6:10], x[6:10])
		wdist = np.linalg.norm(data[0:i,10:13] - x[10:13], axis=1)
		dist = 0.1 * pdist + 0.1 * vdist + 1 * qdist + 0.01 * wdist
		x_near_idx = np.argmin(dist)
		# x_near_idx = np.argmin(np.linalg.norm(data[0:i,0:env.n] - x, axis=1))
		x_near = data[x_near_idx]
		# print(x_near)
		# ids, distances = index.knnQuery(x, k=1)
		# x_near = data[ids[0]]

		# ids, distances = index.knn_query(x, k=1)
		# x_near_idx = int(ids[0][0])
		# x_near = data[x_near_idx]
		# print(x_near_idx, idx)
		# if idx != ids[0][0]:
		# 	idx2 = int(ids[0][0])
		# 	print(idx, ids, distances, np.linalg.norm(data[idx:idx+1] - x, axis=1), np.linalg.norm(data[idx2:idx2+1] - x, axis=1))

		# steer
		best_u = None
		best_dist = None
		best_state = None
		for l in range(num_actions_per_steer):
			# randomly generate a control signal
			u = np.random.uniform(env.a_min, env.a_max)

			# forward propagate
			env.reset(x_near[0:env.n])
			for k in range(num_actions_per_step):
				new_state, _, done, _ = env.step(u)
				if done:
					break

			if not done:
				dist = np.linalg.norm(new_state - x)
				if best_u is None or dist < best_dist:
					best_u = u
					best_state = new_state
					best_dist = dist
		# print(x, x_near, u, new_state)

		# check if state is valid
		if best_u is not None:
			# print(x, x_near, best_u, best_state)
			no_solution_count = 0
			# tree.append(Motion(new_state, m_near))
			data[i,0:env.n] = best_state
			data[i,env.n:env.n+env.m] = best_u
			data[i,env.n+env.m] = x_near_idx
			# index.addDataPoint(i, data[i])
			# index.createIndex(print_progress=False)
			i += 1

			if i % batch_size == 0:
				print(i)
				index.add_items(data[last_index_update_idx:i, 0:env.n])
				last_index_update_idx = i
				# index.addDataPointBatch(data[i-batch_size:i], ids=range(i-batch_size,i+1))
				# index.createIndex(print_progress=False)

			if i % check_goal_iter == 0:
				# ids, distances = index.knn_query(xf, k=1)
				# dist = np.sqrt(distances[0][0])

				pdist = np.linalg.norm(data[0:i-1,0:3] - xf[0:3], axis=1)
				vdist = np.linalg.norm(data[0:i-1,3:6] - xf[3:6], axis=1)
				qdist = rowan.geometry.sym_distance(data[0:i-1,6:10], xf[6:10])
				wdist = np.linalg.norm(data[0:i-1,10:13] - xf[10:13], axis=1)
				dist = 0.1 * pdist + 0.1 * vdist + 1 * qdist + 0.01 * wdist

				print("Distance to goal: ", np.min(dist))
				if np.min(dist) <= eps:
					print("Found goal!")
					break
		else:
			no_solution_count += 1

	with open("rrt.npy", "wb") as f:
		np.save(f, data[0:i], allow_pickle=False)

	index.add_items(data[last_index_update_idx:i, 0:env.n])

	# find the best state with respect to the goal
	xf = param.ref_trajectory[:,-1]
	ids, distances = index.knn_query(xf, k=1)
	idx = int(ids[0][0])
	states = []
	actions = []
	while idx >= 0:
		x_near = data[idx]
		idx = int(x_near[env.n+env.m])
		states.append(x_near[0:env.n])
		actions.append(x_near[env.n:env.n+env.m])

	states.reverse()
	actions.reverse()
	del actions[0]
	actions.append(0)

	print(states)

	result = np.empty(((len(states) - 1) * num_actions_per_step + 1, env.n+env.m))
	for i, (state, action) in enumerate(zip(states, actions)):
		result[i*num_actions_per_step,0:env.n] = state
		result[i*num_actions_per_step,env.n:env.n+env.m] = action
		if i < len(states) - 1:
			env.reset(state)
			for k in range(num_actions_per_step):
				state, _, _, _ = env.step(action)
				result[i*num_actions_per_step+k,0:env.n] = state
				result[i*num_actions_per_step+k,env.n:env.n+env.m] = action

	print(result)
	np.savetxt(param.rrt_fn, result, delimiter=',')

	# # compute reward
	# env.reset(result[0,0:4])
	# for row in result[0:-2]:
	# 	state, _, _, _ = env.step(row[4])
	# 	print(row[4], state)
	# print("Reward: ", env.reward(), " state: ", env.state)

	# # runtime plot
	# fig, ax = plt.subplots()
	# ax.plot(data[:,0], data[:,1], '*')
	# # ax.set_xticklabels(algorithms)
	# # ax.set_ylabel('Runtime [s]')
	# plt.show()

	# fig, ax = plt.subplots()
	# ax.plot(result[:,0])
	# ax.plot(result[:,1])
	# # ax.set_xticklabels(algorithms)
	# # ax.set_ylabel('Runtime [s]')
	# plt.show()

	return states, actions


