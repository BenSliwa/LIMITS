from data.CSV import CSV

# define the training data set 
training = "../examples/mnoA.csv"

# compute amd export the correlation matrix
csv = CSV()
csv.load(training)
csv.computeCorrelationMatrix("results/example_correlation/corr.csv")

# all results are written to results/example_correlation/