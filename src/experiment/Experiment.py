from .CrossValidation import CrossValidation
from .Type import Type
from data.FileHandler import FileHandler
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix
from experiment.Configuration import Configuration
import numpy as np
from enum import Enum

class Experiment:
	def __init__(self, _training, _id="exp_0",  **kwargs):
		self.training = _training
		self.genDataSets = True
		self.seed = 1000
		self.id = _id
		self.verbose = kwargs.get('verbose', True)
		self.resultFolder = "results/" + self.id + "/"		
		self.clear = False

		FileHandler().createFolder("results")
		FileHandler().createFolder(self.resultFolder)
		FileHandler().createFolder(self.resultFolder + "tmp/")


	def regression(self, _models, _folds):
		return self.run(_models, _folds, Type.REGRESSION)


	def classification(self, _models, _folds):
		return self.run(_models, _folds, Type.CLASSIFICATION)


	def run(self, _models, _folds, _type):
		if self.genDataSets:
			csv = CSV()
			csv.load(self.training)
			csv.randomize(self.seed)

			if _type==Type.REGRESSION:
				csv.createFolds(_folds, self.resultFolder + "tmp/")					
			elif _type==Type.CLASSIFICATION:
				classes = csv.stratify(_folds, self.resultFolder + "tmp/")

		R = ResultMatrix()
		for i in range(0, len(_models)):
			model = _models[i] 
			model.modelType = _type
			config = Configuration(self.training, model, _folds)
			config.resultFolder = self.resultFolder 
			config.tmpFolder = self.resultFolder + "tmp/"

			cv = CrossValidation(config, str(i))
			r = cv.run(csv.id, csv.id)
			results = np.hstack([r.data.mean(0), r.data.std(0)])		# TUDO: mean only if size>1 !
			R.add(r.header + [x + "_std" for x in r.header], results)

			if self.verbose:
				if i==0:
					r.printHeader()
				r.printAggregated()

		FileHandler().saveMatrix(R.header, R.data, self.resultFolder + "results.csv")

		if self.clear:
			FileHandler().clearFolder(self.resultFolder + "tmp/")
												
		return R.header, R.data


	def path(self, _file):
		return self.resultFolder + _file


	def tmp(self):
		return self.resultFolder + "tmp/"