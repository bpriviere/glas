import numpy as np
import matplotlib.pyplot as plt
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("file")
	parser.add_argument("--animate", action='store_true')
	args = parser.parse_args()

	# data = np.load(args.file)

	data = np.loadtxt(args.file, delimiter=',', skiprows=1, dtype=np.float32)

	# print(data.dtype)

	# # store in binary format
	# with open("orca.npy", "wb") as f:
	# 	np.save(f, data, allow_pickle=False)

	num_agents = int((data.shape[1] - 1) / 4)
	print(num_agents)

	fig, ax = plt.subplots()
	for i in range(num_agents):
		ax.plot(data[:,i*4+1], data[:,i*4+2])
	plt.show()

	fig, ax = plt.subplots()
	for i in range(num_agents):
		v = np.sqrt(data[:,i*4+3]**2 + data[:,i*4+4]**2)
		ax.plot(v)
	plt.show()

	dt = np.diff(data[:,0])

	fig, ax = plt.subplots()
	for i in range(num_agents):
		v = np.sqrt(data[:,i*4+3]**2 + data[:,i*4+4]**2)
		a = np.diff(v) / dt
		ax.plot(a)
	plt.show()


	if args.animate:
		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		for i in range(num_agents):
			vis["agent"+str(i)].set_object(g.Sphere(1.5))

		while True:
			for row in data:
				t = row[0]
				for i in range(num_agents):
					state = row[i*4+1:i*4+5]
					vis["agent" + str(i)].set_transform(tf.translation_matrix([state[0], state[1], 0]))
				time.sleep(0.01)
