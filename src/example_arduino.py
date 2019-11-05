from weka.models.M5 import M5
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from code.Arduino import Arduino

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = M5()

# perform a 10-fold cross validation
e = Experiment(training, "example_arduino")
e.regression([model], 10)

# export the raw C++ code 
codeFile = "results/" + e.id + "/arduino.cpp"
CodeGenerator().export(training, model, "arduino", codeFile)

# create a dummy Arduino project which executes the model
csv = CSV()
csv.load(training)
attributes = csv.findAttributes(0)

mem = Arduino().run(codeFile, "float", len(attributes)-1)
print(mem)

# all results are written to results/example_arduino/