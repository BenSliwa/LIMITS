from weka.models.M5 import M5
from plot.ResultVisualizer import ResultVisualizer
from experiment.ConvergenceAnalysis import ConvergenceAnalysis

resultFile = "tmp/convergence_m5.txt"
#ConvergenceAnalysis().run("../examples/mnoA.csv", M5(), 100, resultFile)
ResultVisualizer().errorbars([resultFile], "rmse", xlabel='Number of Training Samples', ylabel='RMSE', savePNG='example_model_convergence.png')