from weka.models.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator

# define the training data set and set up the model
training = "../examples/vehicleClassification.csv"
model = RandomForest()
model.trees = 10
model.depth = 10

# perform a 10-fold cross validation
e = Experiment(training, "example_rf")
e.classification([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, "rf", "results/" + e.id + "/rf.cpp")

# all results are written to results/example_rf/


