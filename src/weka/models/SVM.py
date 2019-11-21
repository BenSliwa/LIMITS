from weka.models.LearningModel import LearningModel
from experiment.Type import Type
from data.FileHandler import FileHandler
from data.CSV import CSV
import numpy as np


class SVM(LearningModel):
	def __init__(self, _model):
		super().__init__()
		self.model = _model
		self.modelType = self.model.modelType

	def serialize(self):
		cmd = ""
		if self.modelType==Type.CLASSIFICATION:
			cmd = "weka.classifiers.functions.SMO -C " + str(self.model.config.C) + " -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1"
			cmd += " -K  \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\""
			cmd += " -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\""
		elif self.modelType==Type.REGRESSION:
			cmd = "weka.classifiers.functions.SMOreg -C " + str(self.model.config.C) + " -N 0"
			cmd += " -I \"weka.classifiers.functions.supportVector.RegSMOImproved -T 0.001 -V -P 1.0E-12 -L 0.001 -W 1\""
			cmd += " -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\""

			#cmd = "weka.classifiers.functions.LibSVM -S 4 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -Z -seed 1 -output-debug-info"	

		return cmd


	def parseResults(self, _data, _config, _results):
		weights, classes, offsets = self.parseSVMs(_data)
		# class0,class1,<features>
		features = list(weights[0].keys())
		importance = [float(x) for x in list(weights[0].values())]
		#	TODO: weights are omitted if they are 0 -> integrate known feature vector here		---> Configuration()


	def parseSVMs(self, _data):
		W = []
		C = []
		offsets = []

		if "Classifier for classes:" in _data:
			svms = _data.split("Classifier for classes:")
			for i in range(1, len(svms)):
				classes = svms[i].split("\n")[0].split(",");
				lines = self.extractLines(svms[i], "showing attribute weights, not support vectors", "Number of kernel evaluations")

				weights, offset = self.parseWeights(lines)

				W.append(weights)
				offsets.append(offset)
				C.append([classes[0].strip(" "), classes[1].strip(" ")])
		else:
			lines = self.extractLines(_data, "weights (not support vectors):", "Number of kernel evaluations")

			weights, offset = self.parseWeights(lines)

			W.append(weights)
			offsets.append(offset)

				
		return W, C, offsets


	def parseWeights(self, _lines):
		weights = {}
		offset = 0
		for line in _lines:
			if ")" in line:
				key = line.split(")")[1].strip(" ")
				value = float(line.split("*")[0].replace(" ", "").strip("+"))
				weights[key] = value
			elif "+" in line or "-" in line:
				value = line.replace(" ", "")
				if "+" in value:
					value = value.strip("+")
				offset = value

		return weights, offset


	def initModel(self, _data, _csv, _attributes, _fileIn=""):
		self.model.clear()
		if self.modelType==Type.REGRESSION:
			lines = self.extractLines(_data, "weights (not support vectors):", "Number of kernel evaluations:")
			weights, offset = self.parseWeights(lines)

			self.model.weights = [weights]
			self.model.offsets = [offset]
			self.model.features = list(CSV().createAttributeDict(_attributes[1:]).keys())
			self.model.normedValues = self.model.normalize(_csv, self.model.features)

			x = np.array(_csv.getColumn(0))
			y = x.astype(np.float)
			yRange = max(y)-min(y)
			yMin = min(y)

			code = self.model.generateRegressionCode(_attributes, yMin, yRange)
		else: # classification
			classes = _attributes[0].type.strip("{").strip("}").split(",")

			self.model.weights, self.model.classes, self.model.offsets = self.parseSVMs(_data)
			self.model.features = list(CSV().createAttributeDict(_attributes[1:]).keys())
			self.model.normedValues = self.model.normalize(_csv, self.model.features)


	def exportCode(self, _data, _csv, _attributes, _fileOut, _fileIn="", **kwargs):
		code = ""
		if self.modelType==Type.REGRESSION:
			self.initModel(_data, _csv, _attributes, _fileIn)

			x = np.array(_csv.getColumn(0))
			y = x.astype(np.float)
			yRange = max(y)-min(y)
			yMin = min(y)

			code = self.model.generateRegressionCode(_attributes, yMin, yRange)
		else: # classification
			classes = _attributes[0].type.strip("{").strip("}").split(",")
			self.initModel(_data, _csv,_attributes, _fileIn)
			code = self.model.generateClassificationCode(_attributes, classes)
		
		FileHandler().write(code, _fileOut)


	def toString(self):
		return "SVM"
