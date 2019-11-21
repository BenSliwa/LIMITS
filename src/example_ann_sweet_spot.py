from models.ann.ANN import ANN
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
	
	codeFile = "example_ann_sweet_spot.cpp"
	CodeGenerator().export(_training, _model, codeFile)
	
	if _regression==True:
		resultType = "float"
	else:
		resultType = "const char*"

	mem = []
	platforms = [Arduino(), MSP430(), ESP32()]
	for platform in platforms:
		mem.append(platform.run(codeFile, resultType, lAtt))

	return mem
	

def classificationANN(_training, _layers, _nodes, _file):
	e = Experiment(_training, verbose=False)

	R = ResultMatrix()
	for numLayers in range(1, _layers+1):
		for numNodes in range(1, _nodes+1):
			ann = ANN()
			ann.config.hiddenLayers = []
			for i in range(numLayers):
				ann.config.hiddenLayers.append(numNodes)

			header, result = e.classification([ann], 10)
			mem = computeMemorySize(_training, ann, False)
			header += ["arduino", "msp", "esp"]
			result = np.hstack([result, mem])

			print(["#layers=" + str(numLayers) + "/" + str(_layers) + " nodes=" + str(numNodes) + "/" + str(_nodes) + ' mem=', mem], flush=True)

			R.add(header, result)
	R.save(_file)


def plot(_data, _ax, _title, _xMax):
	for i in range(_data.shape[1]):
		_ax.plot(range(_data.shape[0]), _data)
		_ax.set_xlim(1, _xMax)
		_ax.set_title(_title)
		_ax.set(xlabel='#Nodes on Hidden Layer', ylabel='Program Memory\n Occupation [kB]')


def plotSweetSpot(_example, _file, _layers, _nodes):
	csv = CSV(_file)

	fig, axs = plt.subplots(2,2)
	M = ResultVisualizer().readAsMatrix(_file, "accuracy", _layers, _nodes)
	S = ResultVisualizer().readAsMatrix(_file, "accuracy_std", _layers, _nodes)
	for i in range(_layers):
		y = M[:,i]
		yStd = S[:,i]
		x = range(len(y))

		ax = axs[0,0]
		ax.errorbar(x, y, yerr=yStd, capsize=7)
		ax.set_title('Model Performance')
		ax.set(xlabel='#Nodes on Hidden Layer', ylabel='Accuracy')
	
	plot(ResultVisualizer().readAsMatrix(_file, "msp", _layers, _nodes)/1000, axs[0,1], "MSP430", _nodes)
	plot(ResultVisualizer().readAsMatrix(_file, "arduino", _layers, _nodes)/1000, axs[1,0], "Atmega", _nodes)
	plot(ResultVisualizer().readAsMatrix(_file, "esp", _layers, _nodes)/1000, axs[1,1], "ESP32", _nodes)

	fig.tight_layout()
	fig.set_size_inches(8, 5)
	fig.savefig(_example.path("example_ann_sweet_spot.png"), format="png")

	plt.show()


e = Experiment("", "example_ann_sweet_spot")
layers = 3
nodes = 26

resultFile = e.path("ann_classification_mem.csv")
classificationANN("../examples/vehicleClassification.csv", layers, nodes, resultFile)
plotSweetSpot(resultFile, layers, nodes)