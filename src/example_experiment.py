from weka.models.ANN import ANN
from weka.models.M5 import M5
from weka.models.RandomForest import RandomForest
from weka.models.SVM import SVM
from experiment.Experiment import Experiment
from plot.ResultVisualizer import ResultVisualizer
import matplotlib.pyplot as plt

# define the training data set and set up the model
training = "../examples/mnoA.csv"
models = [ANN(), M5(), RandomForest(), SVM()]

# perform a 10-fold cross validation
e = Experiment(training, "example_experiment")
e.regression(models, 10)

# visualize
files = ["tmp/cv_" + str(i) + ".csv" for i in range(len(models))] 
fig, axs = plt.subplots(2,2)
fig.set_size_inches(8, 5)
xticks = ["ANN", "M5", "Random Forest", "SVM"]
ResultVisualizer().boxplots(files, "r2", xticks,  ylabel='R2', fig=fig, ax=axs[0][0], show=False)
ResultVisualizer().boxplots(files, "mae", xticks,  ylabel='MAE [MBit/s]', fig=fig, ax=axs[0][1], show=False)
ResultVisualizer().boxplots(files, "rmse", xticks,  ylabel='RMSE [MBit/s]', fig=fig, ax=axs[1][0], show=False)
ResultVisualizer().boxplots(files, "training", xticks,  ylabel='Training Time [s]', fig=fig, ax=axs[1][1], savePNG="results/" + e.id + "/"+'example_experiment.png')

# all results are written to results/example_experimment/
