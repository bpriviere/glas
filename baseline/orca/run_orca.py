import subprocess
import numpy as np

if __name__ == '__main__':

	if True:
		for i in range(50):
			# numAgents = int(np.random.uniform(1, 20))
			numAgents = 15
			print(i, numAgents)
			# run process
			subprocess.run(['./orca', '--numAgents', str(numAgents), '--size', str(10)])
			# load file and convert to binary
			data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
			# store in binary format
			with open("orca{}.npy".format(i), "wb") as f:
				np.save(f, data, allow_pickle=False)
			with open("../../../data/singleintegrator/random/orca{}.npy".format(i),"wb") as f2:
				np.save(f2, data, allow_pickle=False)


	else:
		data = np.loadtxt("orca.csv", delimiter=',', skiprows=1, dtype=np.float32)
		# store in binary format
		with open("orca{}.npy".format(1), "wb") as f:
			np.save(f, data, allow_pickle=False)
		with open("../../../data/singleintegrator/ring/orca{}.npy".format(i),"wb") as f2:
			np.save(f2, data, allow_pickle=False)


