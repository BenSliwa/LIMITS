from weka.models.LearningModel import LearningModel
from data.Node import Node
from code.Tree_Model import Tree_Model
from data.FileHandler import FileHandler

# https://stats.stackexchange.com/questions/228724/m5p-interpretations-and-questions
# the code generator currently only works for the regression tree (-R), not the model tree

class M5(LearningModel):
	def __init__(self):
		super().__init__()


	def serialize(self):
		cmd = "weka.classifiers.trees.M5P -M 4.0 -U -R"
		#cmd = "weka.classifiers.trees.M5P -M 4.0"

		return cmd

		
	def parseResults(self, _data, _config, _results):
		""


	def generateGraph(self, _data, _attributes):
		data = _data.split("M5 pruned regression tree:")[1].split("LM num")[0].split("\n")
		#data = _data.split("(using smoothed linear models)")[1].split("LM num")[0].split("\n")

		g = Tree_Model("0")
		g.init(data, _attributes)
		g.linearModels = self.parseLinearModels(_data)

		return g


	def parseLinearModels(self, _data):
		M = {}
		LM = _data.split("LM num:")
		if len(LM)>0:
			for i in range (1, len(LM)):
				lm = self.parseLinearModel(LM[i])
				M["LM" + str(i)] = lm
		return M


	def parseLinearModel(self, _data):
		result = _data.split("Number of Rules")[0].split("=")[1].replace("\n", "").replace(" ", "").replace("\t", "")
		return result


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn="", **kwargs):
		code = ""
		if not "{" in _attributes[0].type: 
			model = self.generateGraph(_data, _attributes)
			code = model.generateGraphCode()
			code = code.replace("tree_0(", "predict(")
			FileHandler().write(code, _fileOut)
		else:
			print("[ERROR] M5 does not support classification")

	def toString(self):
		return "M5"