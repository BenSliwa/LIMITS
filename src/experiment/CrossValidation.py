from weka.Weka import WEKA
from data.ResultMatrix import ResultMatrix
from experiment.ConfusionMatrix import ConfusionMatrix

class CrossValidation:
	def __init__(self, _config, _id=""):
		self.config = _config
		self.id = _id
		

	def run(self, _training, _test):
		weka = WEKA()
		weka.folder = self.config.tmpFolder

		confusion = ConfusionMatrix()
		results = ResultMatrix()
		features = ResultMatrix()
		for i in range(0, self.config.folds):
			training = self.config.tmpFolder + "training_" + _training + "_" + str(i) + ".arff";
			test = self.config.tmpFolder + "test_" + _test + "_" + str(i) + ".arff";

			out = weka.applyModel(self.config.model, training, test, self.id + "_" + str(i))
			weka.parseResults(out, self.config, results, features, confusion)

		#		
		if len(confusion.classes)>0:
			confusion.save(self.config.resultFolder + "confusion_" + self.id + ".csv")
		results.save(self.config.resultFolder + "cv_" + self.id + ".csv")
		if features.header:
			features.save(self.config.resultFolder + "features_" + self.id + ".csv")

		return results

