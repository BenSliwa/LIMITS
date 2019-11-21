from plot.ResultVisualizer import ResultVisualizer
from data.CSV import CSV
from models.randomforest.RandomForest import RandomForest
from experiment.Experiment import Experiment
from code.CodeGenerator import CodeGenerator
from code.CodeEvaluator import CodeEvaluator
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
import numpy as np
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino

from plot.PlotTool import PlotTool
from plot.ResultVisualizer import ResultVisualizer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotSweetSpot(_file, _sX, _sY):
	fig, axs = plt.subplots(2,2)
	PlotTool({"fig":fig, "ax": axs[0][0]}).colorMap(ResultVisualizer().readAsMatrix(_file, "r2", _sX, _sY), "R2")
	PlotTool({"fig":fig, "ax": axs[0][1]}).colorMap(ResultVisualizer().readAsMatrix(_file, "msp", _sX, _sY)/1000, "MSP430 Program Memory Occupation [kB]")
	PlotTool({"fig":fig, "ax": axs[1][0]}).colorMap(ResultVisualizer().readAsMatrix(_file, "arduino", _sX, _sY)/1000, "Atmega Program Memory Occupation [kB]")
	PlotTool({"fig":fig, "ax": axs[1][1]}).colorMap(ResultVisualizer().readAsMatrix(_file, "esp", _sX, _sY)/1000, "ESP32 Program Memory Occupation [kB]")

	for ax in axs.flat:
		ax.set_xticks(range(4,_sX+1,5))
		ax.set_xticklabels(range(5,_sX+1,5))
		ax.set_yticks(range(0,_sY,5))
		ax.set_yticklabels(np.flipud(range(5,_sY+1,5)))
		ax.set(xlabel='Number of Trees', ylabel='Maximum Depth')

	fig.tight_layout()
	fig.set_size_inches(8, 5)
	fig.savefig('example_rf_sweet_spot.png', format='png')
	plt.show()


def computeMemorySize(_training, _model, _resultFolder, _discretization):
	csv = CSV(_training)
	lAtt = len(csv.findAttributes(0))-1
	
	codeFile = "example_rf_sweet_spot.cpp"
	CodeGenerator().export(_training, _model, codeFile, _discretization)
	
	mem = []
	platforms = [Arduino(), MSP430(), ESP32()]
	for platform in platforms:
		mem.append(platform.run(codeFile, "unsigned char", lAtt))

	return mem


def regressionRF(_training, _trees, _depth, _file, _resultFolder, _discretization):
	csv = CSV(training)
	attributes = csv.findAttributes(0)
	
	R = ResultMatrix()
	for numTrees in range(1,_trees+1):
		for depth in range(1,_depth+1):
			rf = RandomForest()
			rf.config.trees = numTrees
			rf.config.depth = depth

			# perform a cross validation to generate the training/test files
			e = Experiment(_training, "example_rf_sweet_spot_disc", verbose=False)
			e.regression([rf], 10)

			# 
			r,c = CodeEvaluator().crossValidation(rf, _training, attributes,  e.tmp(), _discretization)
			result = np.hstack([r.data.mean(0), r.data.std(0)])		
			header = r.header + [x + "_std" for x in r.header]


			mem = computeMemorySize(_training, rf, _resultFolder, _discretization)
			header += ["arduino", "msp", "esp"]
			result = np.hstack([result, mem])

			print(["#trees=" + str(numTrees) + "/" + str(_trees) + " depth=" + str(depth) + "/" + str(_depth) + ' mem=', mem], flush=True)

			R.add(header, result)
	R.save(_file)

#
training = "../examples/mnoA.csv"
resultFolder = "results/example_rf_sweet_spot_disc/"
csv = CSV(training)
attributes = csv.findAttributes(0)
d = None
d = csv.discretizeData()

#
trees = 30
depth = 15
resultFile = resultFolder + "rf_regression_mem_disc.csv"
regressionRF(training, trees, depth, resultFile, resultFolder, d)
plotSweetSpot(resultFile, trees, depth)

