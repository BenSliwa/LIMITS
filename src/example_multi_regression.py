from experiment.MultiExperiment import MultiExperiment
from plot.ResultVisualizer import ResultVisualizer
import matplotlib.pyplot as plt
from models.m5.M5 import M5


# define the training data sets and set up the model
t0 = "../examples/mnoA.csv"
t1 = "../examples/mnoB.csv"
t2 = "../examples/mnoC.csv"
model = M5()

# perform a multi experiment 
m = MultiExperiment("example_multi_regression")
m.run(model, [t0, t1, t2])

# visualize
resultFolder = "results/example_multi_regression/"
files = [resultFolder + x + ".csv" for x in ["mae", "rmse", "r2"]]
ResultVisualizer().colormaps(1, 3, files, ["MAE", "RMSE", "R2"], cmap="Blues", xlabel="Test", ylabel="Training")

