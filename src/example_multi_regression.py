from experiment.MultiExperiment import MultiExperiment
from weka.models.M5 import M5

# define the training data sets and set up the model
t0 = "../examples/mnoA.csv"
t1 = "../examples/mnoB.csv"
t2 = "../examples/mnoC.csv"
model = M5()

# perform a multi experiment 
m = MultiExperiment("example_multi_regression")
m.run(model, [t0, t1, t2])

# all results are written to results/example_multi_regression/


