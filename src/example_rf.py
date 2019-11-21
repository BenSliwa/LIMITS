from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = RandomForest()
model.config.trees = 10
model.config.depth = 5

# perform a 10-fold cross validation
e = Experiment(training, "example_rf")
e.regression([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, e.path("rf.cpp"))

# all results are written to results/example_rf/


