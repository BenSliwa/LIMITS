from experiment.Experiment import Experiment
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix

class ConvergenceAnalysis:

	def __init__(self):
		""


	def run(self, _training, _model, _batchSize, _resultFile):
		csv = CSV()
		csv.load(_training)
		csv.randomize(1000)
		csv.removeIndices()

		R = ResultMatrix()
		for i in range(int(len(csv.data)/_batchSize)):
			c = CSV()
			c.header = csv.header
			c.data = csv.data[0:(i+1)*_batchSize]

			file = "tmp/subset_" + str(i) + ".csv"
			c.save(file)

			header, data = Experiment(file).regression([_model], 10)
			R.add(header, data)

		R.save(_resultFile)