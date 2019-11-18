from weka.models.SVM import SVM
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.ANN_Model import ANN_Model
from data.CSV import CSV
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
import numpy as np
from plot.ResultVisualizer import ResultVisualizer


# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = SVM()

# perform a 10-fold cross validation
e = Experiment(training, "example_svm_feature_importance.py")
e.regression([model], 10)


csv = CSV(training)
attributes = csv.findAttributes(0)
features = csv.header[1:]

# 
for i in range(10):
	training = "tmp/training_mnoA_" + str(i) + ".csv"
	data = "\n".join(FileHandler().read("tmp/raw0_" + str(i) + ".txt"))

	svm = model.buildAbstractModel(data, csv, attributes, training)
	svm.exportWeights(csv.header[1:], "tmp/svm_features_" + str(i) + ".csv")

#
M = ResultMatrix()
for i in range(10):
	csv = CSV("tmp/svm_features_" + str(i) + ".csv")
	D = csv.data[0].split(",")

	M.add(csv.header, np.array([abs(float(x)) for x in D]))
M.normalizeRows()
M.sortByMean()
M.save("tmp/" + e.id + ".csv")

# 
ResultVisualizer().barchart("tmp/" + e.id + ".csv", xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.id+".png")

