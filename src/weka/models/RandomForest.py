from weka.models.LearningModel import LearningModel
from data.FileHandler import FileHandler
from models.randomforest.Tree_Model import Tree_Model
import numpy as np

class RandomForest(LearningModel):
	def __init__(self, _model):
		super().__init__()
		self.model = _model
		self.printTree = True
		self.mdi = True


	def serialize(self):
		cmd = "weka.classifiers.trees.RandomForest -P " + str(self.model.config.bag) + " -I " + str(self.model.config.trees) + " -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth " + str(self.model.config.depth)
		if self.printTree==True:
			cmd += " -print"
		if self.mdi==True:
			cmd += " -attribute-importance"

		cmd += " -num-decimal-places 10"

		return cmd

	def initModel(self, _data, _attributes, _discretization=None):
		self.model.clear()
		self.model.discretization = _discretization
		self.initTrees(_data, _attributes)


	def initTrees(self, _data, _attributes):
		trees = _data.split("RandomTree\n==========")
		l = len(trees)-1
		for i in range(1, l+1):
			items = trees[i].split("Size of the tree")
			tree = items[0].split("\n")
			s = float(items[1].split("\n")[0].strip("Size of the tree : "))

			g = Tree_Model(str(i-1))
			g.discretization = self.model.discretization

			root = g.init(tree, _attributes)
			self.model.trees.append(g)


	def parseResults(self, _data, _config, _results):
		features = []
		importance = []
		if "Attribute importance based on average impurity decrease (and number of nodes using that attribute)" in _data:
			lines = self.extractLines(_data, "Attribute importance based on average impurity decrease (and number of nodes using that attribute)", "Time taken to build model:")
			for line in lines:
				importance.append(float(line.split("(")[0]))
				features.append(line.split(")")[1].strip(" "))
		_results.add(features, np.matrix([importance]))


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn="", **kwargs):
		code = ""
		discretization = kwargs.get("discretization", None)
		if not "{" in _attributes[0].type: 
			self.initModel(_data, _attributes, discretization)
			code = self.model.generateRegressionCode(_attributes)
		else:
			classes = _attributes[0].type.strip("{").strip("}").split(",")
			self.initModel(_data, _attributes, discretization)
			code = self.model.generateClassificationCode(_attributes, classes)
		FileHandler().write(code, _fileOut)


	def toString(self):
		return "RF" + "_t_" + str(self.model.config.trees) + "_d_" + str(self.model.config.depth)
