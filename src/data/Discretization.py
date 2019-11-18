from data.FileHandler import FileHandler
import math

class Discretization:
	def __init__(self):
		self.header = []
		self.min = []
		self.max = []
		self.widths = []
		self.bins = 256
		self.dataType = "unsigned char"


	def save(self, _file):
		M0 = [str(x) for x in self.min]
		M1 = [str(x) for x in self.max]
		W = [str(x) for x in self.widths]
		FileHandler().write(",".join(self.header) + "\n" + ",".join(M0) + "\n" + ",".join(M1) + "\n" + ",".join(W), _file)


	def discretize(self, _key, _value):
		index = self.header.index(_key)
		v = math.floor((_value-self.min[index])/self.widths[index])
		v = min(self.bins-1, v)
		v = max(0, v)

		return v

	def dediscretize(self, _key, _value):
		index = self.header.index(_key)

		return _value * self.widths[index] + self.min[index]	



		