from weka.Weka import WEKA
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
from experiment.ConfusionMatrix import ConfusionMatrix
import numpy as np

class CrossValidation:
	def __init__(self, _config, _id=""):
		self.config = _config
		self.id = _id
		

	def run(self, _training, _test):
		weka = WEKA()
		folder = "tmp/"

		keys = "";
		confusion = ConfusionMatrix()
		R = ResultMatrix()
		F = ResultMatrix()
		for i in range(0, self.config.folds):
			training = folder + "training_" + _training + "_" + str(i) + ".arff";
			test = folder + "test_" + _test + "_" + str(i) + ".arff";

			out = weka.applyModel(self.config.model, training, test, self.id + "_" + str(i))
			self.config.model.parseResults(out, self.config, F)
			
			vals = np.matrix([]);
			if "Detailed Accuracy By Class" in out:
				conf, classification = self.parseClassificafionResult(out)
				confusion.merge(conf)

				keys = list(classification.keys())
				vals =  np.fromiter(classification.values(), dtype=float)
				#print(classification) 
			else:
				reg = self.parseRegressionResult(out)
				keys = list(reg.keys())
				vals =  np.fromiter(reg.values(), dtype=float)

			R.add(keys, vals)


		#		
		if len(confusion.classes)>0:
			confusion.save("tmp/confusion_" + self.id + ".csv")

		R.save("tmp/cv_" + self.id + ".csv")
		F.save("tmp/features_" + self.id + ".csv")

		return R


	def parseClassificafionResult(self, _data):
		classification = {};
		cm = self.parseConfusionMatrix(_data)

		classification["accuracy"], classification["precision"], classification["recall"], classification["f_score"] = cm.calc()

		if "Time taken to build model: " in _data:
			classification["training"] = float(_data.split("Time taken to build model: ")[1].split("seconds")[0])
		if "Time taken to test model on test data:" in _data:
			classification["test"] = float(_data.split("Time taken to test model on test data:")[1].split("seconds")[0])

		return cm, classification


	def parsePredictions(self, _data):
		if "=== Predictions on test data ===" in _data:									# TODO
			""


	def parseConfusionMatrix(self, _data):
		lines = _data.split("=== Confusion Matrix ===")[-1].splitlines()

		cm = ConfusionMatrix()
		cm.data = np.matrix([]);
		for line in lines:
			if "|" in line:
				classType = line.split("= ")[1];
				cm.classes.append(classType)

				line = line.split("|")[0];
				m = np.fromstring(line, dtype=int, sep=' ')

				if cm.data.size==0:
					cm.data = m
				else:
					cm.data = np.vstack([cm.data, m]) 

		return cm


	def parseRegressionResult(self, _data):
		reg = {};
		if "=== Error on test data ===" in _data:
			lines = _data.split("=== Error on test data ===")[1].splitlines()
			for line in lines:
				if "Correlation coefficient" in line:
					corr = float(line.split("Correlation coefficient")[1]);
					reg["r2"] = corr * corr;
				elif "Mean absolute error" in line:
					reg["mae"] = float(line.split("Mean absolute error")[1]);
				elif "Root mean squared error" in line:
					reg["rmse"] = float(line.split("Root mean squared error")[1]);

		if "Time taken to build model: " in _data:
			reg["training"] = float(_data.split("Time taken to build model: ")[1].split("seconds")[0])
		if "Time taken to test model on test data:" in _data:
			reg["test"] = float(_data.split("Time taken to test model on test data:")[1].split("seconds")[0])

		return reg;

