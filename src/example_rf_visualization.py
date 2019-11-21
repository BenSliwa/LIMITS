from models.randomforest.RandomForest import RandomForest
from weka.models.RandomForest import RandomForest as RandomForest_WEKA
from experiment.Experiment import Experiment
from data.CSV import CSV
from code.CodeGenerator import CodeGenerator
from data.FileHandler import FileHandler


# define the training data set and set up the model
training = "../examples/vehicleClassification.csv"
model = RandomForest()
model.config.depth = 7


# perform a 10-fold cross validation
e = Experiment(training, "example_rf")
e.classification([model], 10)


# 
csv = CSV()
csv.load(training)
attributes = csv.findAttributes(0)

data = "\n".join(FileHandler().read(e.tmp() + "raw0_0.txt"))

RandomForest_WEKA(model).initModel(data, attributes)
model.exportEps(model.depth+1, 10, 10, len(attributes)-1)