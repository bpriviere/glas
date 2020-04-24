import torch.nn as nn
import torch


class FeedForward(nn.Module):

	def __init__(self,layers,activation):
		super(FeedForward, self).__init__()
		self.layers = layers
		self.activation = activation

		self.in_dim = layers[0].in_features
		self.out_dim = layers[-1].out_features

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

	def export_to_onnx(self, filename):
		dummy_input = torch.randn(self.in_dim)
		torch.onnx.export(self, dummy_input, "{}.onnx".format(filename), export_params=True, keep_initializers_as_inputs=True)
