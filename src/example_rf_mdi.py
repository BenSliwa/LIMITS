from weka.models.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix
import numpy as np
import matplotlib.pyplot as plt
from plot.PlotTool import PlotTool

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = RandomForest()

# perform a 10-fold cross validation
e = Experiment(training, "example_rf_mdi")
#e.regression([model], 10)

# export the C++ code 
resultFolder = "results/" + e.id
CodeGenerator().export(training, model, "rf", resultFolder + "/rf.cpp")

#
csv = CSV("tmp/features_0.csv")
M = csv.toMatrix()
M.normalizeRows()

Y = np.mean(M.data, axis=0)
S = np.std(M.data, axis=0)

pt = PlotTool()
pt.barchart(Y, S, M.header)
args = {}
args["xlabel"] = "Feature"
args["ylabel"] = "Relative Mean Increase Impurity"
args["savePNG"] = resultFolder + "example_rf_mdi.png"
pt.finalize(args)