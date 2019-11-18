from weka.models.ANN import ANN
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
model = ANN()

# perform a 10-fold cross validation
e = Experiment(training, "example_ann_feature_importance")
e.regression([model], 10)

#
M = ResultMatrix()
csv = CSV(training)
attributes = csv.findAttributes(0)

for i in range(10):
	training = "tmp/training_mnoA_" + str(i) + ".csv"
	data = "\n".join(FileHandler().read("tmp/raw0_" + str(i) + ".txt"))

	ann = model.buildAbstractModel(data, csv, attributes, training)
	M.add(csv.header[1:], ann.computeInputLayerRanking())
M.normalizeRows()
M.sortByMean()
M.save("tmp/ann_features.csv")

#
ResultVisualizer().barchart("tmp/ann_features.csv", xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.id+".png")

