from plot.ResultVisualizer import ResultVisualizer
from data.CSV import CSV
from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.CodeEvaluator import CodeEvaluator
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix


# define the training data set and set up the model
training = "../examples/mnoA.csv"
model = RandomForest()
model.config.trees = 10
model.config.depth = 10

csv = CSV(training)
attributes = csv.findAttributes(0)

# perform a 10-fold cross validation
e = Experiment(training, "example_model_reapplication")
e.regression([model], 10)

#
ce = CodeEvaluator()
R, C = ce.crossValidation(model, training, attributes, e.tmp())
R.printAggregated()

#
ResultVisualizer().scatter([e.tmp()+"predictions_"+str(i)+".csv" for i in range(10)], "prediction", "label", xlabel='Predicted Data Rate [MBit/s]', ylabel='Measured Data Rate [MBit/s', savePNG=e.path("example_model_reapplication.png"))