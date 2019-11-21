from models.Model import Model
from .Config import Config
from .CppSerializer import CppSerializer
from .Tree_Model import Tree_Model
from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.Node import Node
from data.CSV import CSV
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator

class RandomForest(Model):
	def __init__(self):
		super().__init__("RandomForest")
		self.config = Config()
		self.discretization = None
		self.clear()

	def clear(self):
		self.trees = []


	def generateClassificationCode(self, _attributes, _classes):
		return CppSerializer(self).generateClassificationCode(_attributes, _classes)		
	

	def generateRegressionCode(self, _attributes):
		return CppSerializer(self).generateRegressionCode(_attributes)		


	def exportEps(self, _depth, _numX, _numY, _attributes):
		eps = EpsDocument(2000, 2000)

		l = len(self.trees)
		w = eps.width / _numX

		for i in range(0, len(self.trees)):
			g = self.trees[i]
			root = g.root
			
			x = i%_numX
			y = int(i/_numX)
			xOffset = x * w + w/2
			yOffset = y * eps.height / _numY

			g.drawNode(root, 0, xOffset, eps, 0, 0, w, eps.height - yOffset, _depth, _numY, False)

			if i==3:
				eps2 = EpsDocument(400, 400)
				g.drawNode(root, 0, eps2.width/2, eps2, 0, 0, eps2.width, eps2.height, _depth, 1, True)
				eps2.save("tree.eps")

		eps.save("forest.eps")

