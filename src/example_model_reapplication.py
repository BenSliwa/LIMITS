from plot.ResultVisualizer import ResultVisualizer
from data.CSV import CSV
from weka.models.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.CodeEvaluator import CodeEvaluator


# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = RandomForest()
model.trees = 10
model.depth = 10

# perform a 10-fold cross validation
e = Experiment(training, "example_rf")
e.regression([model], 10)
resultFolder = "results/" + e.id + "/"

# export the C++ code 
codeFile = resultFolder + "rf.cpp"
CodeGenerator().export(training, model, "rf", codeFile)

# 
csv = CSV()
csv.load(training)

#
resultFile = resultFolder + "rf_scatter.txt"
CodeEvaluator().regression(codeFile, csv.findAttributes(0), training, resultFile)
ResultVisualizer().scatter(resultFile, "prediction", "label", xlabel='Predicted Data Rate [MBit/s]', ylabel='Measured Data Rate [MBit/s', savePNG=resultFolder+'example_model_reapplication.png')
