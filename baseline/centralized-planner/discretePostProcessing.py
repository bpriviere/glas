#!/usr/bin/env python3
import yaml
import argparse

def discretePostProcessing(input_problem, input_schedule, output):
  with open(input_problem) as input_file:
    data_problem = yaml.load(input_file, Loader=yaml.SafeLoader)

  with open(input_schedule) as input_file:
    data = yaml.load(input_file, Loader=yaml.SafeLoader)

  for k, agent_name in enumerate(data["schedule"]):
    agent = data["schedule"][agent_name]
    path = []
    i = 0
    for i in range(0, len(agent)-1):
      path.append({
        'x': agent[i]['x'],
        'y': agent[i]['y'],
        't': i*2})
      path.append({
        'x': (agent[i]['x'] + agent[i+1]['x'])/2,
        'y': (agent[i]['y'] + agent[i+1]['y'])/2,
        't': i*2+1})
    path.append({
      'x': agent[-1]['x'],
      'y': agent[-1]['y'],
      't': (i+1)*2})
    if len(agent) == 1:
      path.append({
        'x': agent[-1]['x'],
        'y': agent[-1]['y'],
        't': 1})

    # convert to real
    path.insert(0, {
      'x': data_problem["agents"][k]['start_real'][0],
      'y': data_problem["agents"][k]['start_real'][1],
      't': path[0]['t'] - 1})
    path.append({
      'x': data_problem["agents"][k]['goal_real'][0],
      'y': data_problem["agents"][k]['goal_real'][1],
      't': path[-1]['t'] + 1})

    data["schedule"][agent_name] = path

  with open(output, 'w') as output_file:
    yaml.dump(data, output_file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_problem", help="input file containing problem")
  parser.add_argument("input_schedule", help="input file containing schedule")
  parser.add_argument("output", help="output file with post-processed schedule")
  args = parser.parse_args()

  discretePostProcessing(args.input_problem, args.input_schedule, args.output)
