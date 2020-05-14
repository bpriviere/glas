
# standard package
import torch
import gym 
import numpy as np 
import os
import glob
from collections import namedtuple

# my package
from matplotlib.patches import Rectangle, Circle
import utilities as util
import plotter

def run_sim(param, env, controller, initial_state, name=None):
	states = np.empty((len(param.sim_times), env.n))
	actions = np.empty((len(param.sim_times)-1,env.m))
	observations = [] 
	reward = 0 

	env.reset(initial_state)
	states[0] = np.copy(env.s)
	for step, time in enumerate(param.sim_times[:-1]):
		# print('t: {}/{}'.format(time,param.sim_times[-1]))

		state = states[step]
		observation = env.observe()

		action = controller.policy(observation)
		next_state, r, done, _ = env.step(action, compute_reward = False)
		reward += r
		
		states[step + 1] = next_state
		actions[step] = action.flatten()
		observations.append(observation)

		# if done or r < 0:
		# 	break
		if done:
			break

	print('reward: ',reward, name)
	env.close()
	return states, observations, actions, step


def sim(param, env, controllers, initial_state, visualize):

	# environment
	times = param.sim_times
	device = "cpu"

	if initial_state is None:
		initial_state = env.reset()

	# run sim
	SimResult = namedtuple('SimResult', ['states', 'observations', 'actions', 'steps', 'name'])
	
	for name, controller in controllers.items():
		print("Running simulation with " + name)
		print("Initial State: ", initial_state)
		if hasattr(controller, 'policy'):
			result = SimResult._make(run_sim(param, env, controller, initial_state) + (name, ))
		else:
			observations = [] 
			result = SimResult._make((controller.states, observations, controller.actions, controller.steps, name))
		sim_results = []
		sim_results.append(result)

		# plot state space
		if param.env_name in ['SingleIntegrator','SingleIntegratorVelSensing','DoubleIntegrator'] :
			fig,ax = plotter.make_fig()
			ax.set_title('State Space')
			ax.set_aspect('equal')

			for o in env.obstacles:
				ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

			for agent in env.agents:
				
				line = ax.plot(result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)], 
					result.states[0:result.steps,env.agent_idx_to_state_idx(agent.i)+1],alpha=0.5)
				color = line[0].get_color()

				# plot velocity vectors:
				X = []
				Y = []
				U = []
				V = []
				for k in np.arange(0,result.steps,100):
					X.append(result.states[k,env.agent_idx_to_state_idx(agent.i)])
					Y.append(result.states[k,env.agent_idx_to_state_idx(agent.i)+1])
					if param.env_name in ['SingleIntegrator','SingleIntegratorVelSensing']:
						# Singleintegrator: plot actions
						U.append(result.actions[k,2*agent.i+0])
						V.append(result.actions[k,2*agent.i+1])
					elif param.env_name in ['DoubleIntegrator']:
						# doubleintegrator: plot velocities
						U.append(result.states[k,env.agent_idx_to_state_idx(agent.i)+2])
						V.append(result.states[k,env.agent_idx_to_state_idx(agent.i)+3])

				ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=0.5,color=color,width=0.005)
				plotter.plot_circle(result.states[1,env.agent_idx_to_state_idx(agent.i)],
					result.states[1,env.agent_idx_to_state_idx(agent.i)+1],param.r_agent,fig=fig,ax=ax,color=color)
				plotter.plot_square(agent.s_g[0],agent.s_g[1],param.r_agent,angle=45,fig=fig,ax=ax,color=color)

			# draw state for each time step
			robot = 0
			if param.env_name in ['SingleIntegrator']:
				for step in np.arange(0, result.steps, 1000):
					fig,ax = plotter.make_fig()
					ax.set_title('State at t={} for robot={}'.format(times[step], robot))
					ax.set_aspect('equal')

					# plot all obstacles
					for o in env.obstacles:
						ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

					# plot overall trajectory
					line = ax.plot(result.states[0:result.steps,env.agent_idx_to_state_idx(robot)], 
						result.states[0:result.steps,env.agent_idx_to_state_idx(robot)+1],"--")
					color = line[0].get_color()

					# plot current position
					plotter.plot_circle(result.states[step,env.agent_idx_to_state_idx(robot)],
						result.states[step,env.agent_idx_to_state_idx(robot)+1],param.r_agent,fig=fig,ax=ax,color=color)

					# plot current observation
					observation = result.observations[step][robot][0]
					num_neighbors = int(observation[0])
					num_obstacles = int((observation.shape[0]-3 - 2*num_neighbors)/2)

					robot_pos = result.states[step,env.agent_idx_to_state_idx(robot):env.agent_idx_to_state_idx(robot)+2]
					
					idx = 3
					for i in range(num_neighbors):
						pos = observation[idx : idx+2] + robot_pos
						ax.add_patch(Circle(pos, 0.25, facecolor='gray', edgecolor='red', alpha=0.5))
						idx += 2

					for i in range(num_obstacles):
						# pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
						# ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
						pos = observation[idx : idx+2] + robot_pos
						ax.add_patch(Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
						idx += 2

					# plot goal
					goal = observation[1:3] + robot_pos
					ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color=color))

				# 	# import matplotlib.pyplot as plt
				# 	# plt.savefig("test.svg")
				# 	# exit()
			elif param.env_name in ['DoubleIntegrator']:
				for step in np.arange(0, result.steps, 1000):
					fig,ax = plotter.make_fig()
					ax.set_title('State at t={} for robot={}'.format(times[step], robot))
					ax.set_aspect('equal')

					# plot all obstacles
					for o in env.obstacles:
						ax.add_patch(Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

					# plot overall trajectory
					line = ax.plot(result.states[0:result.steps,env.agent_idx_to_state_idx(robot)], 
						result.states[0:result.steps,env.agent_idx_to_state_idx(robot)+1],"--")
					color = line[0].get_color()

					# plot current position
					plotter.plot_circle(result.states[step,env.agent_idx_to_state_idx(robot)],
						result.states[step,env.agent_idx_to_state_idx(robot)+1],param.r_agent,fig=fig,ax=ax,color=color)

					# plot current observation
					observation = result.observations[step][robot][0]
					num_neighbors = int(observation[0])
					num_obstacles = int((observation.shape[0]-5 - 4*num_neighbors)/2)

					robot_pos = result.states[step,env.agent_idx_to_state_idx(robot):env.agent_idx_to_state_idx(robot)+2]

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
						# print(np.linalg.norm(observation[idx+2:idx+4]))
						ax.add_patch(Circle(pos, param.r_agent, facecolor='gray', edgecolor='red', alpha=0.5))
						idx += 4

					for i in range(num_obstacles):
						pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
						ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
						# pos = observation[idx : idx+2] + robot_pos
						# ax.add_patch(Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
						idx += 2

					# plot goal
					goal = observation[1:3] + robot_pos
					ax.add_patch(Rectangle(goal - np.array([0.2,0.2]), 0.4, 0.4, alpha=0.5, color=color))
					X.append(robot_pos[0])
					Y.append(robot_pos[1])
					U.append(observation[3])
					V.append(observation[4])

					# plot velocity vectors
					ax.quiver(X,Y,U,V,angles='xy', scale_units='xy',scale=0.5,color='red',width=0.005)


		# plot time varying states
		if param.env_name in ['SingleIntegrator','DoubleIntegrator']:
			for i_config in range(env.state_dim_per_agent):
				fig,ax = plotter.make_fig()
				ax.set_title(env.states_name[i_config])
				for agent in env.agents:
					for result in sim_results:
						ax.plot(
							times[1:result.steps],
							result.states[1:result.steps,env.agent_idx_to_state_idx(agent.i)+i_config],
							label=result.name)

		# plot time varying actions
		if param.env_name in ['SingleIntegrator','DoubleIntegrator']:
			for i_config in range(env.action_dim_per_agent):
				fig,ax = plotter.make_fig()
				ax.set_title(env.actions_name[i_config])
				for agent in env.agents:
					for result in sim_results:
						ax.plot(
							times[1:result.steps],
							result.actions[1:result.steps,agent.i*env.action_dim_per_agent+i_config],
							label=result.name)

						# 
						if i_config == 5:
							ax.set_yscale('log')

	plotter.save_figs(param.plots_fn)
	plotter.open_figs(param.plots_fn)

	# visualize
	if visualize:
		# plotter.visualize(param, env, states_deeprl)
		env.visualize(sim_results[0].states[0:result.steps],0.1)
