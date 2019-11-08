from weka.models.ANN import ANN
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.ANN_Model import ANN_Model
from data.FileHandler import FileHandler
from data.CSV import CSV

# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = ANN()
model.hiddenLayers = [10, 10]

# perform a 10-fold cross validation
e = Experiment(training, "example_ann")
e.regression([model], 10)

# export the C++ code 
CodeGenerator().export(training, model, "ann", "results/" + e.id + "/ann.cpp")

# all results are written to results/example_ann/




csv = CSV()
csv.load(training)

data = "\n".join(FileHandler().read("tmp/raw0_0.txt"))
annModel = model.generateClassificationModel(data, csv.findAttributes(0), model.hiddenLayers, training)

# save a model visulization
annModel.exportEps('ann_vis.eps')
