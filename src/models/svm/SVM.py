from models.Model import Model
from .Config import Config
from .CppSerializer import CppSerializer
from data.FileHandler import FileHandler
from data.EpsDocument import EpsDocument
from data.CSV import CSV
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator
from experiment.Type import Type
import numpy as np

class SVM(Model):
	def __init__(self):
		super().__init__("SVM")
		self.config = Config()
		self.modelType = Type.REGRESSION
		self.clear()


	def clear(self):
		self.classes = []
		self.features = []
		self.offsets = []
		self.weights = []
		self.normedValues = []
		

	def generateClassificationCode(self, _attributes, _classes):
		return CppSerializer(self).generateClassificationCode(_attributes, _classes)


	def generateRegressionCode(self, _attributes, _yMin, _yRange):
		return CppSerializer(self).generateRegressionCode(_attributes, _yMin, _yRange)


	def getWeights(self, _weights, _header):
		w = []
		for i in range(0, len(_header)):
			key = _header[i]
			v  = 0
			if key in _weights:
				v = _weights[key]
			w.append(str(v))
		return w


	def normalize(self, _csv, _header):
		normedValues = [];
		for j in range(0, len(_header)):
			key = _header[j]
			x = np.array(_csv.getColumn(j+1))
			y = x.astype(np.float)
			r = max(y)-min(y)
			minY = min(y)

			v = ""
			if minY<0:
				v = "(" + key + "+" + str(-minY) + ")/" + str(r)
			else:
				v = "(" + key + "-" + str(minY) + ")/" + str(r)
			normedValues.append(v)

		return normedValues


	def exportWeights(self, _features, _file):
		M = CSV()

		if len(self.classes)>0:
			M.header =  ['class0', 'class1'] + _features
		else:
			M.header = _features


		for c in range(len(self.weights)):
			W = self.weights[c]

			F = [];
			for feature in _features:
				if feature in W:
					F.append(W[feature])
				else:
					F.append(0)

			if len(self.classes)>0:
				M.data.append(','.join(self.classes[c] + [str(x) for x in F]))
			else:
				M.data.append(','.join([str(x) for x in F]))
		M.save(_file)
