import glob
import os
import stats
import numpy as np
import yaml

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


def add_line_plot_agg(pp,result_by_instance,key,title=None, x_label=None, y_label=None, group_by="num_agents",\
	ax=None):
	if ax is None:
		fig,ax = plt.subplots()

	if title:
		ax.set_title(title)
	if x_label:
		ax.set_xlabel(x_label)
	if y_label:
		ax.set_ylabel(y_label)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	# find set of num agent cases
	group_by_set = set()

	for _,results in result_by_instance.items():
		for r in results:
			group_by_set.add(r[group_by])
	x_array = np.array(sorted(list(group_by_set)))

	result_array = np.zeros(( len(solvers), len(x_array), 2))

	for i_s,solver in enumerate(sorted(solvers)):

		for i_a,x in enumerate(x_array):
			
			num_models = set()
			case_count = 0
			curr = dict()
			curr_count = dict()

			for instance,results in result_by_instance.items():
				for r in results:
					if r[group_by] == x and r["solver"] == solver:

						# print(instance)
						# print(r[key])

						if r["num_model"] in num_models:
							curr[r["num_model"]] += r[key]
							curr_count[r["num_model"]] += 1
						else:
							curr[r["num_model"]] = r[key]
							curr_count[r["num_model"]] = 1
							num_models.add(r["num_model"])

			# curr = np.array(list(curr.values())) / num_agent / 10
			curr = np.array([x / curr_count[key] for key, x in curr.items()])
			# print(curr)
			# print(num_models)
			result_array[i_s,i_a,0] = np.mean(curr)
			result_array[i_s,i_a,1] = np.std(curr)

		line = ax.plot(x_array, result_array[i_s,:,0], label=solver)[0]
		ax.fill_between(x_array,
			result_array[i_s,:,0]-result_array[i_s,:,1],
			result_array[i_s,:,0]+result_array[i_s,:,1],
			facecolor=line.get_color(),
			linewidth=1e-3,
			alpha=0.5)

	if key == "percent_agents_success":
		ax.set_ylim([0,100])

	if group_by == "num_agents":
		ax.set_xscale('log')
		# ax.minorticks_off()
	
	ax.set_xticks(np.arange(x_array[0], x_array[-1], 2), True) # set minor ticks
	ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	
	ax.set_xticks(x_array) # set major ticks
	ax.set_xticklabels(x_array)



	if pp is not None:
		plt.legend()
		pp.savefig(fig)
		plt.close(fig)



def add_scatter(pp, result_by_instance, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	width = 0.8 / len(solvers)

	for k, solver in enumerate(sorted(solvers)):
		idx = 0
		x = []
		y = []
		for instance in sorted(result_by_instance):
			results = result_by_instance[instance]
			for r in results:
				if r["solver"] == solver:
					x.append(idx)
					y.append(r[key])
			idx += 1
		# ax.scatter(x,y,label=solver)
		ax.bar(np.array(x)+k*width, y, width, label=solver)


	ax.set_xticks(np.arange(len(result_by_instance)))
	# ax.set_xticklabels([instance for instance, _ in result_by_instance.items()])

	plt.legend()

	pp.savefig(fig)
	plt.close(fig)


def add_bar_agg(pp, result_by_instance, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	for k, solver in enumerate(sorted(solvers)):

		agg = 0
		for _, results in result_by_instance.items():
			for r in results:
				if r["solver"] == solver:
					agg += r[key]
		ax.bar(k, agg)

	ax.set_xticks(np.arange(len(solvers)))
	ax.set_xticklabels([solver for solver in sorted(solvers)])

	pp.savefig(fig)
	plt.close(fig)

def add_bar_agg_succeeded_agents(pp, result_by_instance, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	# find set of solvers
	solvers = set()
	for _, results in result_by_instance.items():
		for r in results:
			solvers.add(r["solver"])

	# x = []
	# y = []
	for k, solver in enumerate(sorted(solvers)):

		agg = 0
		for _, results in result_by_instance.items():
			# compute the set of agents that succeeded in all cases
			agents_succeeded = results[0]["agents_succeeded"]
			for r in results:
				agents_succeeded = agents_succeeded & r["agents_succeeded"]

			# aggregate the ky only for the agents in the set
			for r in results:
				if r["solver"] == solver:
					for a in agents_succeeded:
						agg += r[key][a]
		ax.bar(k, agg)
		# x.append(k)
		# y.append(agg)
	# print(y)
	# ax.bar(x, y)

	ax.set_xticks(np.arange(len(solvers)))
	ax.set_xticklabels([solver for solver in sorted(solvers)])

	pp.savefig(fig)
	plt.close(fig)


def add_bar_chart(pp, results, key, title):
	fig, ax = plt.subplots()
	ax.set_title(title)

	y_pos = np.arange(len(results))
	ax.bar(y_pos, [d[key] for d in results])
	ax.set_xticks(y_pos)
	ax.set_xticklabels([r["solver"] for r in results])
	# ax.set_ylabel(key)

	pp.savefig(fig)
	plt.close(fig)


if __name__ == '__main__':

	result_by_instance = dict()

	# files = list(glob.glob("**/*obst6_agents4_ex000*.npy", recursive=True))
	# files = list(glob.glob("orca/*obst6*.npy", recursive=True))
	# files = list(glob.glob("**/*obst6_agents30_*.npy", recursive=True))

	# files = list(glob.glob("orca/*obst6_agents10_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst6_agents20_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst6_agents30_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst6_agents40_*.npy", recursive=True))
	# files = sorted(files)

	# files = list(glob.glob("orca/*obst9_agents10_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst9_agents20_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst9_agents30_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst9_agents40_*.npy", recursive=True))
	# files = sorted(files)

	# files = list(glob.glob("orca/*obst12_agents10_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst12_agents20_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst12_agents30_*.npy", recursive=True))
	# files.extend(glob.glob("orca/*obst12_agents40_*.npy", recursive=True))
	# files = sorted(files)


	agents_lst = [64] #[2,4,8,16,32]
	obst_lst = [6] #,9,12] #int(map_size[0] * map_size[1] * 0.1)

	# solvers = ['il', 'ilAPF', 'ad', 'adAPF']
	# solvers = ['orca', 'ilvsAPF', 'ilAPF']
	# solvers = ['orca', 'ilAPF', 'central', 'barrier']
	solvers = ['current']

	files = []
	for solver in solvers:
		for obst in obst_lst:
			for agent in agents_lst:
				files.extend( glob.glob("{}/*obst{}_agents{}_*.npy".format(solver,obst,agent), recursive=True))
	for file in files:
		instance = os.path.splitext(os.path.basename(file))[0]
		map_filename = "instances/{}.yaml".format(instance)
		result = stats.stats(map_filename, file)

		if instance in result_by_instance:
			result_by_instance[instance].append(result)
		else:
			result_by_instance[instance] = [result]
		# print(file, solver, instance)

	pp = PdfPages("results.pdf")

	add_line_plot_agg(pp, result_by_instance, "num_agents_success", "# robots success")
	add_line_plot_agg(pp, result_by_instance, "control_effort_sum", "control effort")

	add_bar_agg(pp, result_by_instance, "num_agents_success", "# robots success")
	# add_bar_agg_succeeded_agents(pp, result_by_instance, "control_effort", "total control effort")
	# add_scatter(pp, result_by_instance, "percent_agents_reached_goal", "% reached goal")
	add_scatter(pp, result_by_instance, "num_collisions", "# collisions")

	# pp.close()
	# exit()
	

	for instance in sorted(result_by_instance):
		print(instance)
		results = result_by_instance[instance]

		add_bar_chart(pp, results, "percent_agents_reached_goal", instance + " (% reached goal)")
		add_bar_chart(pp, results, "num_collisions", instance + " (# collisions)")

		map_filename = "instances/{}.yaml".format(instance)
		with open(map_filename) as map_file:
			map_data = yaml.load(map_file, Loader=yaml.SafeLoader)

		for r in results:
			print("state space" + r["solver"])
			fig, ax = plt.subplots()
			ax.set_title("State Space " + r["solver"])
			ax.set_aspect('equal')

			for o in map_data["map"]["obstacles"]:
				ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))
			for x in range(-1,map_data["map"]["dimensions"][0]+1):
				ax.add_patch(Rectangle([x,-1], 1.0, 1.0, facecolor='gray', alpha=0.5))
				ax.add_patch(Rectangle([x,map_data["map"]["dimensions"][1]], 1.0, 1.0, facecolor='gray', alpha=0.5))
			for y in range(map_data["map"]["dimensions"][0]):
				ax.add_patch(Rectangle([-1,y], 1.0, 1.0, facecolor='gray', alpha=0.5))
				ax.add_patch(Rectangle([map_data["map"]["dimensions"][0],y], 1.0, 1.0, facecolor='gray', alpha=0.5))

			data = np.load("{}/{}.npy".format(r["solver"], instance))
			num_agents = len(map_data["agents"])
			dt = data[1,0] - data[0,0]
			for i in range(num_agents):
				# plot trajectory
				line = ax.plot(data[:,1+i*4], data[:,1+i*4+1],alpha=0.5)
				color = line[0].get_color()

				# plot velocity vectors:
				X = []
				Y = []
				U = []
				V = []
				for k in np.arange(0,data.shape[0], int(5.0 / dt)):
					X.append(data[k,1+i*4+0])
					Y.append(data[k,1+i*4+1])
					U.append(data[k,1+i*4+2])
					V.append(data[k,1+i*4+3])

				ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=0.5,color=color,width=0.005)

				# plot start and goal
				start = np.array(map_data["agents"][i]["start"])
				goal = np.array(map_data["agents"][i]["goal"])
				ax.add_patch(Circle(start + np.array([0.5,0.5]), 0.2, alpha=0.5, color=color))
				ax.add_patch(Rectangle(goal + np.array([0.3,0.3]), 0.4, 0.4, alpha=0.5, color=color))

			pp.savefig(fig)
			plt.close(fig)



	pp.close()
