from weka.models.ANN import ANN
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.ANN_Model import ANN_Model
from data.CSV import CSV
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
import numpy as np
import matplotlib.pyplot as plt
from plot.PlotTool import PlotTool

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

Y = np.mean(M.data, axis=0)
S = np.std(M.data, axis=0)

# 
pt = PlotTool()
pt.barchart(Y, S, M.header)
args = {}
args["xlabel"] = "Feature"
args["ylabel"] = "Relative Feature Importance"
args["savePNG"] = e.id + ".png"
pt.finalize(args)

