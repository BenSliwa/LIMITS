from models.m5.M5 import M5
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from code.ESP32 import ESP32

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = M5()

# perform a 10-fold cross validation
e = Experiment(training, "example_esp")
e.regression([model], 10)

# export the raw C++ code 
codeFile = e.path("esp.cpp")
CodeGenerator().export(training, model, codeFile)

# create a dummy ESP32 project which executes the model
csv = CSV()
csv.load(training)
attributes = csv.findAttributes(0)

mem = ESP32().run(codeFile, "float", len(attributes)-1)
print(mem)

# all results are written to results/example_esp/