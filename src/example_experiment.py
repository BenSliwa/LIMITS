from weka.models.ANN import ANN
from weka.models.M5 import M5
from weka.models.RandomForest import RandomForest
from weka.models.SVM import SVM
from experiment.Experiment import Experiment
from plot.ResultVisualizer import ResultVisualizer


# define the training data set and set up the model
training = "../examples/mnoA.csv"
models = [ANN(), M5(), RandomForest(), SVM()]

# perform a 10-fold cross validation
e = Experiment(training, "example_experiment")
e.regression(models, 10)
resultFolder = "results/" + e.id + "/"

# visualize
files = ["tmp/cv_" + str(i) + ".csv" for i in range(len(models))] 
ResultVisualizer().boxplots(files, "r2", ["ANN", "M5", "Random Forest", "SVM"],  ylabel='R2', savePNG=resultFolder+'example_experiment.png')

# all results are written to results/example_experimment/
