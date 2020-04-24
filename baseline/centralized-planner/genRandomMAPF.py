"""
Script to create random MAPF instance
"""

import os
import collections
import random
import copy
import yaml
import numpy as np 


r_agent = 0.2

def reachable(map_size, start, goal, obstacles):
    visited = set()
    stack = [tuple(start)]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            if vertex == tuple(goal):
                return True
            visited.add(vertex)
            for delta in [[1,0], [-1,0], [0,1], [0,-1]]:
                pos = (vertex[0] + delta[0],vertex[1] + delta[1])
                if pos[0] >= 0 and pos[0] < map_size[0] and pos[1] >= 0 and pos[1] < map_size[1] and pos not in obstacles:
                    stack.append(pos)
    return False

def check_collision(agent_loc,obstacles):
    x,y = agent_loc[:]

    collision = False
    for o in obstacles:
        if x > o[0]-0.5 and x < o[0]+0.5 and \
            y  > o[1]-0.5 and y < o[1]+0.5:
            collision = True
            break
    return collision


def check_obst_and_agents_collision(agent_loc,agents,obstacles,start):
    x,y = agent_loc[:]

    x_min = x - r_agent
    x_max = x + r_agent
    y_min = y - r_agent
    y_max = y + r_agent

    collision = False
    for o in obstacles:
        if x_max > o[0]-0.5 and x_min < o[0]+0.5 and \
            y_max  > o[1]-0.5 and y_min < o[1]+0.5:
            return True

    for agent_j in agents:

        if start:
            p_j = agent_j.start[0]
        else:
            p_j = agent_j.goal[0]

        if x_max > p_j[0] - r_agent and x_min < p_j[0] + r_agent and \
            y_max  > p_j[1] - r_agent and y_min < p_j[1] + r_agent:
            return True

    return False 


def interesting(map_size, start, goal, obstacles):

    line = np.linspace(np.array(start),np.array(goal),50)

    if np.linalg.norm(line[0,:]-line[-1,:]) < 2.0:
        return False

    # return True

    for point in line:
        if check_collision(point,obstacles):
            return True

    return False


def randAgents1(map_size, num_agents, num_groups, num_obstacles):
    

    while True:
        locations = [(x, y) for x in range(0, map_size[0]) for y in range(0, map_size[1])]

        random.shuffle(locations)

        #
        Group = collections.namedtuple('Group', 'start goal')
        groups = []
        obstacles = []

        # assign obstacles
        for agentIdx in range(0, num_obstacles):
            location = locations[0]
            obstacles.append(location)
            del locations[0]

        locationsE = copy.deepcopy(locations) #list(locations)
        random.shuffle(locationsE)

        # different number of agents; fixed agents per group
        for groupIdx in range(0, num_groups):
            group = Group(start=[], goal=[])
            groups.append(group)

        success = np.zeros(num_agents,dtype=bool)
        for agentIdx in range(0, num_agents):
            groupIdx = agentIdx % num_groups

            for _ in range(5000):

                if continuous:

                    # print(len(groups))

                    locationS = (np.random.uniform()*(map_size[0]-1)+r_agent,np.random.uniform()*(map_size[1]-1)-r_agent)
                    while check_obst_and_agents_collision(locationS, groups[0:groupIdx], obstacles,1):
                        locationS = (np.random.uniform()*(map_size[0]-1)+r_agent,np.random.uniform()*(map_size[1]-1)-r_agent)

                    locationE = (np.random.uniform()*(map_size[0]-1),np.random.uniform()*(map_size[1]-1))
                    while check_obst_and_agents_collision(locationE, groups[0:groupIdx],obstacles,0):
                        locationE = (np.random.uniform()*(map_size[0]-1)+r_agent,np.random.uniform()*(map_size[1]-1)-r_agent)

                    if interesting(map_size, locationS,locationE, obstacles):
                        groups[groupIdx].start.append(locationS)
                        groups[groupIdx].goal.append(locationE)
                        success[agentIdx] = True
                        break
                    else:
                        # print("not reachable!")
                        random.shuffle(locations)
                        random.shuffle(locationsE)
                        # try again...

                else:
                    locationS = locations[0]
                    locationE = locationsE[0]

                    if reachable(map_size, locationS, locationE, obstacles) and \
                       interesting(map_size, locationS,locationE, obstacles):
                    # if interesting(map_size, locationS,locationE, obstacles):
                        groups[groupIdx].start.append(locationS)
                        groups[groupIdx].goal.append(locationE)
                        success[agentIdx] = True
                        del locations[0]
                        del locationsE[0]
                        # print("reachable!")
                        break
                    else:
                        # print("not reachable!")
                        random.shuffle(locations)
                        random.shuffle(locationsE)
                        # try again...                    
    
        if success.all():
            return groups, obstacles

def writeFile(obstacles, map_size, groups, file_name):
    data = dict()
    data["map"] = dict()
    data["map"]["dimensions"] = map_size
    data["map"]["obstacles"] = [list(o) for o in obstacles]
    data["agents"] = []
    i = 0
    for group in groups:
        for agentIdx in range(0, len(group.start)):
            agent = dict()
            agent["name"] = "agent" + str(i)
            if continuous:
                agent["start"] = (np.array(group.start[agentIdx])).tolist()
                agent["goal"] = (np.array(group.goal[agentIdx])).tolist()
            else:
                slack = 0.5 - r_agent
                agent["start"] = (np.array(group.start[agentIdx])+np.random.uniform(-slack, slack, 2)).tolist()
                agent["goal"] = (np.array(group.goal[agentIdx])+np.random.uniform(-slack, slack, 2)).tolist()
            i += 1
            data["agents"].append(agent)
    with open(file_name, "w") as f:
        yaml.dump(data, f, indent=4, default_flow_style=None)

if __name__ == "__main__":

    # map_size = [32, 32]
    continuous = False
    map_size = [8, 8]
    agents_lst = [8] #[2,4,8,16,32] #,64] #[4,10,20,30] [40,50,100] # np.arange(50,51,10,dtype=int) # [35] 
    obst_lst = [12] #int(map_size[0] * map_size[1] * 0.1)
    cases = range(1000)
    # cases = []

    for num_agents in agents_lst:
        print('num_agents: ', num_agents)
        for num_obstacles in obst_lst:
            print('   num_obstacles: ', num_obstacles)

            for i in cases:
              print('      ',i)
              groups, obstacles = randAgents1(map_size, num_agents, num_agents, num_obstacles)
              writeFile(obstacles, map_size, groups, "map_{0}by{1}_obst{2:02}_agents{3:03}_ex{4:06}.yaml".format(
                  map_size[0],
                  map_size[1],
                  num_obstacles,
                  num_agents,
                  i))