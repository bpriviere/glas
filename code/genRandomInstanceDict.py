"""
Script to create random MAPF instance
"""

import os
import collections
import random
import copy
import yaml
import numpy as np 


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
        if x  > o[0]-0.5 and x < o[0]+0.5 and \
            y  > o[1]-0.5 and y  < o[1]+0.5:
            collision = True
            break
    return collision

def interesting(map_size, start, goal, obstacles):

    line = np.linspace(np.array(start),np.array(goal),50)

    if np.linalg.norm(line[0,:]-line[-1,:]) < 2.0:
        return False

    # for o in obstacles:
    #     print(o)
    #     print(map_size)
    #     exit()
    #     if o[0] == 0 or o[1] == 0 or o[0] == map_size[0]-1 or o[1] == map_size[1]-1:
    #         return False

    return True



    # for point in line:
    #     if check_collision(point,obstacles):
            # return True

    # return False


def randAgents1(map_size, num_agents, num_groups, num_obstacles):
    # locations = [(x, y) for x in range(1, map_size[0]-1) for y in range(1, map_size[1]-1)]
    locations = [(x, y) for x in range(map_size[0]) for y in range(map_size[1])]

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

    # locations.extend([(0, y) for y in range(map_size[1])])
    # locations.extend([(map_size[0]-1, y) for y in range(map_size[1])])
    # locations.extend([(x, 0) for x in range(1,map_size[0]-1)])
    # locations.extend([(x, map_size[1]-1) for x in range(1,map_size[0]-1)])

    locationsE = copy.deepcopy(locations) #list(locations)
    random.shuffle(locationsE)

    # different number of agents; fixed agents per group
    for groupIdx in range(0, num_groups):
        group = Group(start=[], goal=[])
        groups.append(group)

    for agentIdx in range(0, num_agents):
        groupIdx = agentIdx % num_groups

        while True:
            locationS = locations[0]
            locationE = locationsE[0]

            if reachable(map_size, locationS, locationE, obstacles) and \
               interesting(map_size, locationS,locationE, obstacles):
                groups[groupIdx].start.append(locationS)
                groups[groupIdx].goal.append(locationE)
                del locations[0]
                del locationsE[0]
                # print("reachable!")
                break
            else:
                # print("not reachable!")
                random.shuffle(locations)
                random.shuffle(locationsE)
                # try again...

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
            agent["start"] = (np.array(group.start[agentIdx])+np.random.uniform(-0.3, 0.3, 2)).tolist()
            agent["goal"] = (np.array(group.goal[agentIdx])+np.random.uniform(-0.3, 0.3, 2)).tolist()
            i += 1
            data["agents"].append(agent)
    with open(file_name, "w") as f:
        yaml.dump(data, f, indent=4, default_flow_style=None)


def get_random_instance(num_agents,num_obstacles):

    # map_size = [32, 32]
    map_size = [8, 8]

    groups, obstacles = randAgents1(map_size, num_agents, num_agents, num_obstacles)

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
            agent["start"] = (np.array(group.start[agentIdx])+np.random.uniform(-0.3, 0.3, 2)).tolist()
            agent["goal"] = (np.array(group.goal[agentIdx])+np.random.uniform(-0.3, 0.3, 2)).tolist()
            i += 1
            data["agents"].append(agent)

    return data