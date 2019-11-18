from weka.models.LearningModel import LearningModel
from code.Forest_Model import Forest_Model
from code.Tree_Model import Tree_Model
from data.FileHandler import FileHandler
import numpy as np

class RandomForest(LearningModel):
	def __init__(self):
		super().__init__()
		self.trees = 100;
		self.depth = 0;
		self.bag = 100
		self.printTree = True
		self.mdi = True


	def serialize(self):
		cmd = "weka.classifiers.trees.RandomForest -P " + str(self.bag) + " -I " + str(self.trees) + " -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -depth " + str(self.depth)
		if self.printTree==True:
			cmd += " -print"
		if self.mdi==True:
			cmd += " -attribute-importance"

		cmd += " -num-decimal-places 10"

		return cmd


	def parseResults(self, _data, _config, _results):
		features = []
		importance = []
		if "Attribute importance based on average impurity decrease (and number of nodes using that attribute)" in _data:
			lines = self.extractLines(_data, "Attribute importance based on average impurity decrease (and number of nodes using that attribute)", "Time taken to build model:")
			for line in lines:
				importance.append(float(line.split("(")[0]))
				features.append(line.split(")")[1].strip(" "))
		_results.add(features, np.matrix([importance]))


	def generateModel(self, _data, _attributes, _discretization=None):
		model = Forest_Model()
		model.discretization = _discretization
		model.parse(_data, _attributes)

		return model


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn="", **kwargs):
		code = ""
		discretization = kwargs.get("discretization", None)
		if not "{" in _attributes[0].type: 
			model = self.generateModel(_data, _attributes, discretization)
			code = model.generateRegressionCode(_attributes)
		else:
			classes = _attributes[0].type.strip("{").strip("}").split(",")
			model = self.generateModel(_data, _attributes, discretization)
			code = model.generateClassificationCode(_attributes, classes)
		FileHandler().write(code, _fileOut)


	def toString(self):
		return "RF" + "_t_" + str(self.trees) + "_d_" + str(self.depth)
