#!/usr/bin/env python3
import yaml
import argparse

def discretePreProcessing(input_fn, output_fn):
  with open(input_fn) as input_file:
    data = yaml.load(input_file, Loader=yaml.SafeLoader)

  for agent in data["agents"]:
    agent['start_real'] = agent['start']
    agent['goal_real'] = agent['goal']
    agent['start'] = [round(v) for v in agent['start']]
    agent['goal'] = [round(v) for v in agent['goal']]

  with open(output_fn, 'w') as output_file:
    yaml.dump(data, output_file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="input file containing schedule")
  parser.add_argument("output", help="output file with pre-processed schedule")
  args = parser.parse_args()

  discretePreProcessing(args.input, args.output)
