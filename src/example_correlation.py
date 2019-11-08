from data.CSV import CSV
from plot.ResultVisualizer import ResultVisualizer


# define the training data set 
training = "../examples/mnoA.csv"

# compute amd export the correlation matrix
csv = CSV()
csv.load(training)

resultFolder = "results/example_correlation/"
resultFile = resultFolder + "corr.csv"
csv.computeCorrelationMatrix(resultFile)

ResultVisualizer().colorMap(resultFile, savePNG=resultFolder+'example_correlation.png')

# all results are written to results/example_correlation/