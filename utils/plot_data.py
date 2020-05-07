import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import yaml
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("map", help="input file containing map")
	parser.add_argument("schedule")
	parser.add_argument("--animate", action='store_true')
	args = parser.parse_args()

	if os.path.splitext(args.schedule)[1] == ".npy":
		data = np.load(args.schedule)
	elif os.path.splitext(args.schedule)[1] == ".csv":
		data = np.loadtxt(args.schedule, delimiter=',', skiprows=1, dtype=np.float32)
	else:
		raise Exception("Unknown file extension!")

	with open(args.map) as map_file:
		map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

	# print(data.dtype)

	# # store in binary format
	# with open("orca.npy", "wb") as f:
	# 	np.save(f, data, allow_pickle=False)

	num_agents = int((data.shape[1] - 1) / 4)

	fig, (ax0,ax1,ax2) = plt.subplots(1,3)
	ax0.set_aspect('equal', adjustable='box')
	for i in range(num_agents):
		ax0.plot(data[:,i*4+1], data[:,i*4+2])
	for o in map_data["map"]["obstacles"]:
		ax0.add_patch(plt.Rectangle(o, 1.0, 1.0))
	ax0.set_xlim(0, map_data["map"]["dimensions"][0])
	ax0.set_xlabel("x [m]")
	ax0.set_ylim(0, map_data["map"]["dimensions"][1])
	ax0.set_label("y [m]")
	ax0.set_title("X/Y Plot")

	ax1.set_title("Velocity")
	for i in range(num_agents):
		v = np.sqrt(data[:,i*4+3]**2 + data[:,i*4+4]**2)
		ax1.plot(data[:,0],v)

	ax2.set_title("Acceleration")
	dt = np.diff(data[:,0])
	for i in range(num_agents):
		v = np.sqrt(data[:,i*4+3]**2 + data[:,i*4+4]**2)
		a = np.diff(v) / dt
		ax2.plot(data[0:-1,0],a)
	plt.show()

	if args.animate:
		import meshcat
		import meshcat.geometry as g
		import meshcat.transformations as tf
		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		for i in range(num_agents):
			vis["agent"+str(i)].set_object(g.Sphere(0.2))

		for i, o in enumerate(map_data["map"]["obstacles"]):
			vis["obstacles"+str(i)].set_object(g.Box([1.0, 1.0, 0.2]))
			print(o)
			vis["obstacles"+str(i)].set_transform(tf.translation_matrix(np.array([o[0]+0.5, o[1]+0.5, 0])))

		while True:
			for k in np.arange(0,data.shape[0],10):
				t = data[k,0]
				for i in range(num_agents):
					state = data[k,i*4+1:i*4+5]
					vis["agent" + str(i)].set_transform(tf.translation_matrix([state[0], state[1], 0]))
				time.sleep(0.01)
