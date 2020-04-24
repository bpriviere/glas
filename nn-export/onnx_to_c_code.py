import argparse
import onnx
from onnx import numpy_helper
from onnx import optimizer
from onnx import shape_inference
import matplotlib.pyplot as plt
import numpy as np

# converts a numpy array to a C-style string
def arr2str(a):
	return np.array2string(a,
		separator=',',
		floatmode='unique',
		threshold = 1e6,
		max_line_width = 1e6).replace("[","{").replace("]","}").replace("\n","")



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="Onxx input file")
	parser.add_argument("name", help="export name")
	args = parser.parse_args()

	onnx_model = onnx.load(args.input)

	# print('The model is:\n{}'.format(onnx_model))

	# Check the model
	onnx.checker.check_model(onnx_model)
	# print('The model is checked!')

	# export static arrays
	weights = onnx_model.graph.initializer

	# export C struct definition
	print("struct neuralNetworkFF_" + args.name)
	print("{")
	for w in weights:
		name = w.name.replace(".", "_")
		a = numpy_helper.to_array(w).T

		res = "	float " + name
		for s in a.shape:
			res += "[" + str(s) + "]"
		res += ";"
		print(res)
	print("};")

	# export actual variable holding the data from the network
	print("static const struct neuralNetworkFF_{} {} = {{".format(args.name, args.name))
	for w in weights:
		name = w.name.replace(".", "_")
		a = numpy_helper.to_array(w).T
		res = "." + name
		res += " = " + arr2str(a) + ","
		print(res)
	print("};")
