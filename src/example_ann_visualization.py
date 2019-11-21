from models.ann.ANN import ANN
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.FileHandler import FileHandler
from data.CSV import CSV

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = ANN()
model.hiddenLayers = [10, 10]

# perform a 10-fold cross validation
e = Experiment(training, "example_ann_visualization")
e.regression([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, e.path("ann.cpp"))
model.exportEps(e.path("ann_vis.eps"))
