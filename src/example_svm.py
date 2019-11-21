from models.svm.SVM import SVM
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator

# define the training data set and set up the model
training = "../examples/vehicleClassification.csv"
model = SVM()

# perform a 10-fold cross validation
e = Experiment(training, "example_svm")
e.classification([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, e.path("svm.cpp"))

# all results are written to results/example_svm/


