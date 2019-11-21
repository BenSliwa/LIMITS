from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from data.CSV import CSV
from plot.ResultVisualizer import ResultVisualizer

# define the training data set and set up the model
training = "../examples/mnoA.csv"
csv = CSV(training)
model = RandomForest()

# perform a 10-fold cross validation
e = Experiment(training, "example_feature_reduction")
e.regression([model], 10)
CSV(e.path("cv_0.csv")).save(e.path("subset_0.csv"))
xTicks = ["None"]

# obtain a feature ranking
M = CSV(e.path("features_0.csv")).toMatrix()
M.normalizeRows()
M.sortByMean()

# sequentially remove the least important feature from the training data and retrain the model
subset = e.path("subset.csv")
for i in range(len(M.header)-1):
	key = M.header[-1]
	M.header = M.header[0:-1]
	csv.removeColumnWithKey(key)
	csv.save(subset)

	e = Experiment(subset, "example_feature_reduction")
	e.regression([model], 10)
	CSV(e.path("cv_0.csv")).save(e.path("subset_" + str(i+1) + ".csv"))
	xTicks.append(key)

#
files = [e.path("subset_" + str(i) + ".csv") for i in range(len(xTicks))]
ResultVisualizer().boxplots(files, "r2", xTicks,  xlabel='Sequentially Removed Features', ylabel='R2', savePNG=e.path("example_feature_reduction.png"))