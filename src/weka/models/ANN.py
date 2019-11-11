from weka.models.LearningModel import LearningModel
from data.FileHandler import FileHandler
from code.ANN_Model import ANN_Model
from data.CSV import CSV
import numpy as np
from experiment.Experiment import Type

class ANN_Node:
	def __init__(self):
		self.threshold = 0
		self.weights = []


class ANN(LearningModel):
	def __init__(self):
		super().__init__()
		self.learningRate = 0.3;
		self.momentum = 0.2;
		self.epochs = 500;
		self.hiddenLayers = [10, 10]


	def serialize(self):
		cmd = "weka.classifiers.functions.MultilayerPerceptron" 
		cmd += " -L " + str(self.learningRate) + " -M " + str(self.momentum) + " -N " + str(self.epochs) + " -V 0 -S 0 -E 20"
		cmd += " -H " + ",".join(map(str, self.hiddenLayers))
		#cmd += " -H " + str(self.hiddenLayers)
		return cmd

	def parseResults(self, _data, _config, _results):
		""


	def buildAbstractModel(self, _data, _csv, _attributes, _fileIn=""):
		model = []
		if not "{" in _attributes[0].type: 
			model = self.generateRegressionModel(_data, self.hiddenLayers, _fileIn)
		else:
			model = self.generateClassificationModel(_data, _attributes, self.hiddenLayers, _fileIn)
		return model


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn=""):	# IMPORTANT: _fileIn is the training data set of the current fold NOT the global training data set
		model = self.buildAbstractModel(_data, _csv, _attributes, _fileIn)
		model.generateCode(_fileOut)
		
		return model


	def generateClassificationModel(self, _data, _attributes, _layers, _training):
		# determine the mapping of output nodes to classes (the first N nodes correspond to the resulting classes)
		classes = self.extractClasses(_attributes)

		N, O = self.parseNodes(_data, len(classes))
		L = self.parseLayers(_layers, N, O)

		matrices = [];
		for i in range(0, len(L)):
			matrices.append(self.generateWeightMatrix(L[i]))
		matrices.append(self.generateWeightMatrix(O))

		model = ANN_Model()
		model.modelType = Type.CLASSIFICATION
		model.layers = matrices
		model.L = L

		csv = CSV()
		csv.load(_training)
		model.inputLayer = csv.header[1:]
		model.training = _training
		model.outputLayer = classes

		return model


	def generateRegressionModel(self, _data, _layers, _training):
		N, O = self.parseNodes(_data, 1)
		L = self.parseLayers(_layers, N, O)

		# generate the weight and threshold matrices
		matrices = [];
		for i in range(0, len(L)):
			matrices.append(self.generateWeightMatrix(L[i]))
		matrices.append(self.generateWeightMatrix(O))

		model = ANN_Model()
		model.modelType = Type.REGRESSION
		model.layers = matrices
		model.L = L

		csv = CSV()
		csv.load(_training)
		model.inputLayer = csv.header[1:]
		model.training = _training
		model.outputLayer.append(csv.header[0])	

		return model


	def parseNodes(self, _data, _numOut):
		lines = self.extractLines(_data, "=== Classifier model (full training set) ===", "Time taken to build model")
		
		nodes = []
		node = ""
		data = ""
		for line in lines:
			if "Linear" in line or "Sigmoid" in line:
				if data!="":
					nodes.append(self.parseNode(data))
					data = ""
				node = line

			elif "Class" in line:
				nodes.append(self.parseNode(data))
				break
			elif line!="" and not "Inputs    Weights" in line:
				data += line + "\n"

		out = nodes[0:_numOut]
		nodes = nodes[_numOut:]

		return nodes, out


	def parseNode(self, _data):
		node = ANN_Node()

		lines = _data.split("\n")
		for line in lines:
			line = " ".join(line.split())

			if "Threshold" in line:
				node.threshold = float(line.replace("Threshold", "").strip(" "))
			elif line!="":
				items = line.split()
				node.weights.append(float(items[2]))

		return node


	def parseLayers(self, _layers, _nodes, _out):
		layerIndex = 0
		layers = []
		layer = []
		for node in _nodes:
			layer.append(node)
			if len(layer)==_layers[layerIndex]:
				layers.append(layer)
				layer = []
				layerIndex += 1

		return layers


	def generateWeightMatrix(self, _layer):
		f = len(_layer[0].weights)
		W = np.zeros((f, len(_layer)))
		T = []
		for y in range(0, f):
			for x in range(0, len(_layer)):
				W[y][x] = _layer[x].weights[y]

		for node in _layer:
			T.append(str(node.threshold))

		return W, T


	def toString(self):
		return "ANN"
