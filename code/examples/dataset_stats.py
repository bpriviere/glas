import numpy as np
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':

	data = np.empty((0,5))
	for file in glob.glob('../models/CartPole/dataset_rl/*.csv'):
		file_data = np.loadtxt(file, delimiter=',')
		data = np.vstack([data, file_data[0:-1]])

	print(data.shape)

	fig, ax = plt.subplots()
	ax.plot(data[:,0], data[:,1], '.')
	# ax.set_xticklabels(algorithms)
	# ax.set_ylabel('Runtime [s]')
	plt.show()

	fig, ax = plt.subplots()
	ax.plot(data[:,2], data[:,3], '.')
	# ax.set_xticklabels(algorithms)
	# ax.set_ylabel('Runtime [s]')
	plt.show()
