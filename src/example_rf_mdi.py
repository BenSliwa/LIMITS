from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix
import numpy as np
import matplotlib.pyplot as plt
from plot.PlotTool import PlotTool
from plot.ResultVisualizer import ResultVisualizer

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = RandomForest()
model.config.trees = 10
model.config.depth = 5

# perform a 10-fold cross validation
e = Experiment(training, "example_rf_mdi")
e.regression([model], 10)

#
M = CSV(e.path("features_0.csv")).toMatrix()
M.normalizeRows()
M.sortByMean()
M.save(e.path("rf_features.csv"))

#
ResultVisualizer().barchart(e.path("rf_features.csv"), xlabel="Feature", ylabel="Relative Feature Importance", savePNG=e.path(e.id+".png"))