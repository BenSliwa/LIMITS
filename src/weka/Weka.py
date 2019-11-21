import subprocess
from data.FileHandler import FileHandler
from settings.Settings import Settings
from models.Model import Model
from weka.models.ANN import ANN
from weka.models.M5 import M5 
from weka.models.RandomForest import RandomForest 
from weka.models.SVM import SVM 
from experiment.ConfusionMatrix import ConfusionMatrix
import numpy as np

class WEKA:
	def __init__(self):
		self.predictions = False 
		self.modelInterface = None
		self.folder = "tmp/"

	def run(self, _cmd, _id):
		result = subprocess.run(_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		err = result.stderr.decode()
		out = result.stdout.decode()

		if out:
			FileHandler().write(out, self.folder + "raw" + _id + ".txt")
		if err:
			FileHandler().write(err, self.folder + "err" + _id + ".txt")

		return out


	def applyModel(self, _model, _training, _test, _id):
		cmd = "java -cp " + Settings().wekaPath + " -Xmx512m "
		cmd += self.generateWekaModel(_model).serialize()
		cmd += " -c 1 -t " + _training + " -T " + _test

		if self.predictions:
			cmd += " -classifications weka.classifiers.evaluation.output.prediction.CSV"

		return self.run(cmd, _id)
		

	def train(self, _model, _training, _id):
		cmd = "java -cp " + Settings().wekaPath + " -Xmx512m "
		cmd += self.generateWekaModel(_model).serialize()
		cmd += " -c 1 -t " + _training + " -no-cv"

		return self.run(cmd, _id)


	def generateWekaModel(self, _model):
		if _model.modelName=="ANN":
			self.modelInterface = ANN(_model)
		elif _model.modelName=="M5":
			self.modelInterface = M5(_model)
		elif _model.modelName=="RandomForest":
			self.modelInterface = RandomForest(_model)
		elif _model.modelName=="SVM":
			self.modelInterface = SVM(_model)
		return self.modelInterface


	def parseResults(self, _out, _config, _results, _features, _confusion):
		keys = "";
		vals = np.matrix([]);
		if "Detailed Accuracy By Class" in _out:
			conf, classification = self.parseClassificafionResult(_out)
			_confusion.merge(conf)

			keys = list(classification.keys())
			vals =  np.fromiter(classification.values(), dtype=float)
		else:
			reg = self.parseRegressionResult(_out)
			keys = list(reg.keys())
			vals =  np.fromiter(reg.values(), dtype=float)

		_results.add(keys, vals)
		self.modelInterface.parseResults(_out, _config, _features)

		
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

