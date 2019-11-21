from models.randomforest.RandomForest import RandomForest
from plot.ResultVisualizer import ResultVisualizer
from experiment.ConvergenceAnalysis import ConvergenceAnalysis

e = ConvergenceAnalysis("example_model_convergence")
e.run("../examples/mnoA.csv", RandomForest(), 100, e.resultFolder+"convergence_rf.txt")
ResultVisualizer().errorbars([e.resultFolder+"convergence_rf.txt"], "rmse", xlabel='Number of Training Samples', ylabel='RMSE', savePNG=e.resultFolder+'example_model_convergence.png')