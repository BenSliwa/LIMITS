from experiment.CrossValidation import CrossValidation
from experiment.Configuration import Configuration
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
from data.CSV import CSV
import numpy as np


class MultiExperiment:
	def __init__(self, _id="multi_0"):
		self.id = _id		
		self.seed = 1000

		FileHandler().createFolder("results")
		FileHandler().createFolder("results/" + self.id)
		FileHandler().createFolder("results/" + self.id + "/tmp")
		FileHandler().clearFolder(self.id)


	def run(self, _model, _files):
		R = {}
		l = len(_files)
		folds = 10
		for y in range(l):
			training = _files[y]
			for x in range(l):
				cfg = Configuration(training, _model, folds)
				cfg.resultFolder = "results/" + self.id + "/"
				cfg.tmpFolder = cfg.resultFolder + "tmp/"

				test = _files[x]
				
				csvA = CSV(training)	
				csvA.randomize(self.seed)
				csvA.createFolds(folds, cfg.tmpFolder)		

				csvB = CSV(test)
				csvB.randomize(self.seed)
				csvB.createFolds(folds, cfg.tmpFolder)		

				cv = CrossValidation(cfg)
				cv.model = _model
				cv.folds = folds

				cv.id = str(y) + "_" + str(x)
				r = cv.run(csvA.id, csvB.id)

				results = np.hstack([r.data.mean(0), r.data.std(0)])		# TUDO: mean only if size>1 !

				# init the result matrices
				if len(R)==0:
					for key in r.header:
						R[key] = ResultMatrix([FileHandler().generateId(file) for file in _files], np.zeros((l, l)))
						R[key + "_std"] = ResultMatrix([FileHandler().generateId(file) for file in _files], np.zeros((l, l)))

				# update the result matrices
				for i in range(len(r.header)):
					R[r.header[i]].data[y][x] = results[i] 	
					R[r.header[i] + "_std"].data[y][x] = results[i]

		for key in R.keys():
			R[key].save("results/" + self.id + "/" + key + ".csv")