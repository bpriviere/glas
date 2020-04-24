
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages


# some plotting stuff 
plt.rcParams.update({'font.size': 18})
plt.rcParams['lines.linewidth'] = 4


def show():
	plt.show()


def save_figs(filename):
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])

def subplots():
	return plt.subplots()


def plot(T,X,title=None,fig=None,ax=None,label=None,color=None):
	
	if fig is None or ax is None:
		fig, ax = plt.subplots()
	
	line, = ax.plot(T,X)
	if label is not None:
		line.set_label(label)
		ax.legend()
	if color is not None:
		line.set_color(color)
	if title is not None:
		ax.set_title(title)
	return fig, ax


def make_fig(axlim = None):
	fig, ax = plt.subplots()
	if axlim is None:
		# ax.set_aspect('equal')
		pass
	else:
		ax.set_xlim(-axlim[0],axlim[0])
		ax.set_ylim(-axlim[1],axlim[1])
		ax.set_autoscalex_on(False)
		ax.set_autoscaley_on(False)
		ax.set_aspect('equal')
	return fig, ax



def plot_circle(x,y,r,fig=None,ax=None,title=None,label=None,color=None):
	if fig is None or ax is None:
		fig, ax = plt.subplots()

	if False:
		zorder=3
		circle = patches.Circle((x,y),radius=r, zorder = zorder)
	else:
		circle = patches.Circle((x,y),radius=r)
		
	if color is not None:
		circle.set_color(color)
	if label is not None:
		circle.set_label(label)
	if title is not None:
		ax.set_title(title)

	ax.add_artist(circle)
	return fig,ax

def plot_square(x,y,r,angle=None,fig=None,ax=None,title=None,label=None,color=None):
	if fig is None or ax is None:
		fig, ax = plt.subplots()

	if False:
		zorder=3
		rect = patches.Rectangle((x,y),height=2*r,width=2*r,zorder = zorder)
	else:
		if angle is not None: 
			rect = patches.Rectangle((x,y-np.sqrt(2)/2*r),height=2*r,width=2*r,angle=angle)
		else:
			rect = patches.Rectangle((x-r,y-r),height=2*r,width=2*r)

	if color is not None:
		rect.set_color(color)
	if label is not None:
		rect.set_label(label)
	if title is not None:
		ax.set_title(title)

	ax.add_artist(rect)
	return fig,ax	
	

def plot_barrier_fnc(env):
	
	def make_barrier_plot(rsafe,rsense,title=None):
		fig,ax = make_fig()
		d = np.linspace(1.001*rsafe,1.1*rsense,100)
		h = d - rsafe
		b = np.multiply(d < rsense, env.b_gamma*np.power(h,-1*env.b_exph))
		ax.plot(d,b)
		ax.axvline(rsafe,color='red')
		ax.axvline(rsense,color='orange')
		ax.set_title(title)
		ax.set_yscale('log')

	rsafe_agents = env.r_agent*2
	rsense_agents = env.r_comm 
	rsafe_obstacles = env.r_agent + env.r_obstacle
	rsense_obstacles = env.r_obs_sense

	make_barrier_plot(rsafe_agents, rsense_agents, title='Agent Barrier')
	make_barrier_plot(rsafe_obstacles, rsense_obstacles, title='Obstacle Barrier')


def plot_ss(env,states):
	fig,ax = env.render()
	ax.plot(states[:,0],states[:,1],linestyle='dashed')


# def debug_plot_batch(batch):
	# batch = [o_lst,c_lst,r_lst,op_lst,p_lst,d_lst,s_lst,sp_lst]






