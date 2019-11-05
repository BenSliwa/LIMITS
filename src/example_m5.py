from weka.models.M5 import M5
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = M5()

# perform a 10-fold cross validation
e = Experiment(training, "example_m5")
e.regression([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, "m5", "results/" + e.id + "/m5.cpp")

# all results are written to results/example_m5/


