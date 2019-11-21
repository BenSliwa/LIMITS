from models.Model import Model
from .CppSerializer import CppSerializer
from models.ann.Config import Config
from data.EpsDocument import EpsDocument
from experiment.Type import Type
import numpy as np


class ANN(Model):
	def __init__(self):
		super().__init__("ANN")
		self.config = Config()
		self.training = ""
		self.modelType = Type.REGRESSION
		self.clear()

	def clear(self):
		self.inputLayerKeys = [] 
		self.outputLayerKeys = [] 
		self.weights = []
		self.thresholds = []
		self.L = [] 

	def generateCode(self, _file):
		CppSerializer(self).generateCode(_file)		


	def exportEps(self, _file):
		eps = EpsDocument(900, 600)

		layers = [len(self.inputLayerKeys)];
		for l in self.weights:
			layers.append(len(l[1]))
		#layers.append(len(self.outputLayerKeys))

		w = eps.width / len(layers)
		lastLayer = []
		h = 40 # eps height / max layer size
		for i in range(0, len(layers)):
			l = layers[i]
			x = i*w + 50
			yOffset = (eps.height - h*l)/2 
			layer = []
			for j in range(0, l): 
				y = yOffset + j * h
				y += h/2

				eps.drawCircle(x, y, 5, "")
				layer.append([x, y])

				if i==0: # inputLayerKeys
					eps.text(self.inputLayerKeys[j], 12, x - w/10, y, 0, "-1", "-0.5")
				elif i==len(layers)-1: # output layer
					eps.text(self.outputLayerKeys[j], 12, x + w/10, y, 0, "0", "-0.5")

				for p in lastLayer:
					eps.drawLine(x, y, p[0], p[1])
			lastLayer = layer

		eps.save(_file)


	def computeInputLayerRanking(self):
		W = np.abs(self.weights[0])
		S = np.sum(W)
		V = np.sum(W, axis=1) / S

		return V