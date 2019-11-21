from models.svm.SVM import SVM
from weka.models.SVM import SVM as SVM_WEKA
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
import numpy as np
from plot.ResultVisualizer import ResultVisualizer


# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = SVM()

# perform a 10-fold cross validation
e = Experiment(training, "example_svm_feature_importance")
e.regression([model], 10)


csv = CSV(training)
attributes = csv.findAttributes(0)
features = csv.header[1:]

# 
for i in range(10):
	training = e.tmp() + "training_mnoA_" + str(i) + ".csv"
	data = "\n".join(FileHandler().read(e.tmp() + "raw0_" + str(i) + ".txt"))

	SVM_WEKA(model).initModel(data, csv, attributes, training)
	model.exportWeights(csv.header[1:], e.tmp() + "svm_features_" + str(i) + ".csv")

#
M = ResultMatrix()
for i in range(10):
	csv = CSV(e.tmp() + "svm_features_" + str(i) + ".csv")
	D = csv.data[0].split(",")

	M.add(csv.header, np.array([abs(float(x)) for x in D]))
M.normalizeRows()
M.sortByMean()
M.save(e.path(e.id+".csv"))

# 
ResultVisualizer().barchart(e.path(e.id+".csv"), xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.path(e.id+".png"))

