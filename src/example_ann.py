from models.ann.ANN import ANN
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = ANN()
model.config.hiddenLayers = [5, 5]

# perform a 10-fold cross validation
e = Experiment(training, "example_ann")
e.regression([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, e.path("ann.cpp"))

# all results are written to results/example_ann/



