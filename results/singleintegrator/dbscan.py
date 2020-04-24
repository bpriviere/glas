import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


data = np.load("../../code/batch_train_nn0_no2.npy")

num_neighbors = int(data[0,0])
num_obstacles = int((data.shape[1]-5 - 4*num_neighbors - 2)/2)
print("num_neighbors", num_neighbors, "num_obstacles", num_obstacles)

# delete column 1 (num_neighbors), 3 and 4 (current velocity)
X = np.delete(data,[0,3,4],1)
print(X.shape)
# X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1e-1, min_samples=1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# print(set(labels))

pp = PdfPages("dbscan.pdf")

for k in set(labels):
	class_member_mask = (labels == k)
	xy = data[class_member_mask & core_samples_mask]
	if len(xy) > 1:
		print(k)
		# bla = xy
		# for i in range(bla.shape[0]):
			# xy = bla[i:i+1,:]

		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_title("Cluster {} ({} points)".format(k, len(xy)))
		ax.set_xlim([-2,2])
		ax.set_ylim([-2,2])

		for i in range(num_obstacles):
			idx = 5+4*num_neighbors+2*i
			o = np.mean(xy[:,idx:idx+2], axis=0)
			ax.add_patch(Rectangle(o - np.array([0.5,0.5]), 1.0, 1.0, facecolor='gray', edgecolor='black', alpha=0.5))
			for row in xy:
				o = row[idx:idx+2]
				ax.add_patch(Rectangle(o - np.array([0.5,0.5]), 1.0, 1.0, facecolor='gray', alpha=0.1))

		# for row in xy:
		# 	o = np.array(row[5:6])
		# 	ax.add_patch(Rectangle(o - np.array([0.5,0.5]), 1.0, 1.0, facecolor='gray', alpha=0.5))

		goal = np.mean(xy[:,1:3], axis=0)
		ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, edgecolor='black'))
		for row in xy:
			goal = row[1:3]
			ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.1))

		ax.add_patch(Circle(np.array([0,0]), 0.2, alpha=0.5))

		ax.scatter(xy[:,-2], xy[:,-1])

		pp.savefig(fig)
		plt.close(fig)

pp.close()
