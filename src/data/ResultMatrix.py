from data.FileHandler import FileHandler
import numpy as np

class ResultMatrix:
	def __init__(self, _header=[], _data=np.matrix([])):
		self.header = _header
		self.data = _data
		self.colWidth = 17


	def add(self, _header, _data):
		if np.prod(self.data.shape)==0:
			self.header = _header
			self.data = _data
		else:
			self.data = np.vstack([self.data, _data])


	def printAggregated(self):		# TODO: check size -> for 1, return the current values, not the aggregation
		mean = self.data.mean(0);
		std = self.data.std(0);

		output = []
		for i in range(len(self.header)):
			output.append("{:.3f}".format(mean[i]) + "+/-" + "{:.3f}".format(std[i]))
		print("".join(word.ljust(self.colWidth) for word in output), flush=True)


	def printHeader(self):
		print("".join(word.ljust(self.colWidth) for word in self.header[0:len(self.header)]), flush=True)


	def save(self, _file):
		FileHandler().saveMatrix(self.header, self.data, _file)