from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.CodeEvaluator import CodeEvaluator
from data.CSV import CSV

# define the training data set and set up the model
training = "../examples/mnoA.csv"
training = "../examples/vehicleClassification.csv"

csv = CSV(training)
attributes = csv.findAttributes(0)
d = csv.discretizeData()


model = RandomForest()
model.config.trees = 10
model.config.depth = 5

# perform a 10-fold cross validation
e = Experiment(training, "example_rf_disc")
e.classification([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, e.path("rf.cpp"), d)

#
ce = CodeEvaluator()
R, C = ce.crossValidation(model, training, attributes, e.tmp(), d)
R.printAggregated()

# all results are written to results/example_rf_disc/


