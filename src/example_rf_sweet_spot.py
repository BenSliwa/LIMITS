from weka.models.RandomForest import RandomForest
from code.CodeGenerator import CodeGenerator
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino
from data.FileHandler import FileHandler
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix
from experiment.Experiment import Type, Experiment
from plot.PlotTool import PlotTool
from plot.ResultVisualizer import ResultVisualizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def computeMemorySize(_training, _model, _regression):
	csv = CSV(_training)
	lAtt = len(csv.findAttributes(0))-1
	
	codeFile = "example_rf_sweet_spot.cpp"
	CodeGenerator().export(_training, _model, "codeFile", codeFile)
	
	if _regression==True:
		resultType = "float"
	else:
		resultType = "const char*"

	mem = []
	platforms = [Arduino(), MSP430(), ESP32()]
	for platform in platforms:
		mem.append(platform.run(codeFile, resultType, lAtt))

	return mem


def regressionRF(_training, _trees, _depth, _file):
	e = Experiment(_training, verbose=False)

	R = ResultMatrix()
	for numTrees in range(1,_trees+1):
		for depth in range(1,_depth+1):
			rf = RandomForest()
			rf.trees = numTrees
			rf.depth = depth

			header, result = e.regression([rf], 10)
			mem = computeMemorySize(_training, rf, True)
			header += ["arduino", "msp", "esp"]
			result = np.hstack([result, mem])

			print(["#trees=" + str(numTrees) + "/" + str(_trees) + " depth=" + str(depth) + "/" + str(_depth) + ' mem=', mem], flush=True)

			R.add(header, result)
	R.save(_file)


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

#
trees = 30
depth = 15
resultFile = "tmp/rf_regression_mem.csv"
regressionRF("../examples/mnoA.csv", trees, depth, resultFile)
plotSweetSpot(resultFile, trees, depth)