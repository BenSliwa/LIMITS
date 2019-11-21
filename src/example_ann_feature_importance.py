from models.ann.ANN import ANN
from weka.models.ANN import ANN as ANN_WEKA
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
import numpy as np
from plot.ResultVisualizer import ResultVisualizer

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = ANN()

# perform a 10-fold cross validation
e = Experiment(training, "example_ann_feature_importance")
e.regression([model], 10)

#
M = ResultMatrix()
csv = CSV(training)
attributes = csv.findAttributes(0)

for i in range(10):
	training = e.tmp() + "training_mnoA_" + str(i) + ".csv"
	data = "\n".join(FileHandler().read(e.tmp() + "raw0_" + str(i) + ".txt"))

	ANN_WEKA(model).initModel(data, csv, attributes, training)
	M.add(csv.header[1:], model.computeInputLayerRanking())
M.normalizeRows()
M.sortByMean()
M.save(e.path("ann_features.csv"))

#
ResultVisualizer().barchart(e.path("ann_features.csv"), xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.path(e.id+".png"))

