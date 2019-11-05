from experiment.CrossValidation import CrossValidation
from experiment.Experiment import Type, Experiment
from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.CSV import CSV
from code.Forest_Model import Forest_Model
from code.Compiler import Compiler
from plot.PlotTool import PlotTool
from code.CodeGenerator import CodeGenerator
from code.ANN_Model import ANN_Model
from weka.Weka import WEKA
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino
from data.ResultMatrix import ResultMatrix
import numpy as np


# TODO: Detect regression / classification


class ModelRecommender:
	def __init__(self):
		""

	def run(self, _training, _models, _platforms):
		R = ResultMatrix()
		M = [];
		for model in _models:
			# run the cross validation to compute the model performance
			M.append(model.toString())
			e = Experiment(_training)
			header, result = e.regression([model], 10)
			R.add(header, result)

			# train with the global training data and export code
			training_arff = "tmp/recommend.arff"

			csv = CSV()
			csv.load(_training)
			csv.convertToARFF(training_arff, False)
			attributes = csv.findAttributes(0)
			lAtt = len(attributes)-1
			WEKA().train(model, training_arff, "0")
			
			data = "\n".join(FileHandler().read("tmp/raw0.txt"))
			codeFile = "recommend.c"

			model.exportCode(data, csv, attributes, codeFile)

			# complile platform-specific code
			for platform in _platforms:
				""





				#print(model.toString() + " : " + platform.toString())
		print(R.header, R.data)
		print(M)
