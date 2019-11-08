from weka.models.RandomForest import RandomForest
from experiment.Experiment import Experiment
from data.CSV import CSV
from code.CodeGenerator import CodeGenerator
from code.Forest_Model import Forest_Model
from data.FileHandler import FileHandler


# define the training data set and set up the model
training = "../examples/vehicleClassification.csv"
model = RandomForest()
model.depth = 7


# perform a 10-fold cross validation
e = Experiment(training, "example_rf")
e.classification([model], 10)


# 
csv = CSV()
csv.load(training)
attributes = csv.findAttributes(0)

data = "\n".join(FileHandler().read("tmp/raw0_0.txt"))

rf = model.generateModel(data, attributes)
rf.exportEps(model.depth+1, 10, 10, len(attributes)-1)