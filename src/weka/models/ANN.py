from weka.models.LearningModel import LearningModel
from models.ann.Node import Node
from experiment.Type import Type
from data.FileHandler import FileHandler
from data.CSV import CSV
import numpy as np

class ANN(LearningModel):
	def __init__(self, _model):
		super().__init__()
		self.model = _model


	def serialize(self):
		cmd = "weka.classifiers.functions.MultilayerPerceptron" 
		cmd += " -L " + str(self.model.config.learningRate) + " -M " + str(self.model.config.momentum) + " -N " + str(self.model.config.epochs) + " -V 0 -S 0 -E 20"
		cmd += " -H " + ",".join(map(str, self.model.config.hiddenLayers))

		return cmd


	def parseResults(self, _data, _config, _results):
		""


	def initModel(self, _data, _csv, _attributes, _fileIn=""):
		self.model.clear()

		N = []
		O = []
		L = []
		csv = CSV()
		csv.load(_fileIn)
		self.model.inputLayerKeys = csv.header[1:]
		self.model.training = _fileIn

		if not "{" in _attributes[0].type: 
			N, O = self.parseNodes(_data, 1)
			L = self.parseLayers(self.model.config.hiddenLayers, N, O)
			self.model.modelType = Type.REGRESSION		
			self.model.outputLayerKeys.append(csv.header[0])	
		else:
			classes = self.extractClasses(_attributes)
			N, O = self.parseNodes(_data, len(classes))
			L = self.parseLayers(self.model.config.hiddenLayers, N, O)
			self.model.modelType = Type.CLASSIFICATION
			self.model.outputLayerKeys = classes

		for i in range(len(L)):
			W, T = self.generateWeightMatrix(L[i])
			self.model.weights.append(W)
			self.model.thresholds.append(T)

		W, T = self.generateWeightMatrix(O)
		self.model.weights.append(W)
		self.model.thresholds.append(T)
		self.model.L = L


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn="", **kwargs):	# IMPORTANT: _fileIn is the training data set of the current fold NOT the global training data set
		self.initModel(_data, _csv, _attributes, _fileIn)
		self.model.generateCode(_fileOut)
		

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
		node = Node()

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
