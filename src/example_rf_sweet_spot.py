from weka.models.ANN import ANN
from weka.models.M5 import M5
from weka.models.SVM import SVM
from weka.models.RandomForest import RandomForest
from experiment.CrossValidation import CrossValidation
from experiment.Experiment import Type, Experiment
from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.CSV import CSV
from code.Forest_Model import Forest_Model
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator
from code.ANN_Model import ANN_Model
from weka.Weka import WEKA
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino
import numpy as np
from data.ResultMatrix import ResultMatrix
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def computeMemorySize(_training, _model, _regression):
	csv = CSV()
	csv.load(_training)
	attributes = csv.findAttributes(0)
	lAtt = len(attributes)-1
	
	codeFile = "results/example_rf_sweet_spot/codeFile.cpp"
	codeFile = "example_rf_sweet_spot.cpp"
	CodeGenerator().export(_training, _model, "codeFile", codeFile)
	
	if _regression==True:
		resultType = "float"
	else:
		resultType = "const char*"

	arduino = Arduino().run(codeFile, resultType, lAtt)
	msp = MSP430().run(codeFile, "ml_test", resultType, lAtt)
	esp = ESP32().run(codeFile, resultType, lAtt)

	mem = np.array([arduino, msp, esp])


	return mem


def regressionRF():
	training = "../examples/mnoA.csv"
	e = Experiment(training)

	results = np.matrix([])
	cnt = 0
	for numTrees in range(1,31):
		for depth in range(1,16):
			# MODEL
			rf = RandomForest()
			rf.trees = numTrees
			rf.depth = depth
			rf.printTree = True

			# ACC
			header, result = e.regression([rf], 10)

			# MEM
			mem = computeMemorySize(training, rf, True)
			header += ["arduino", "msp", "esp"]
			result = np.hstack([result, mem])

			print(mem, flush=True)

			if cnt==0:
				results = result
			else:
				results = np.vstack([results, result])
			cnt += 1

	FileHandler().saveMatrix(header, results, "tmp/rf_regression_mem_.csv")

def grid(_data, _ax, _xlabel):
	_data = np.flipud(_data)
	im = _ax.imshow(_data, cmap="jet")

	divider = make_axes_locatable(_ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	clb = plt.colorbar(im, cax=cax)

	_ax.set_title(_xlabel)


def getMatrix(_csv, _key, _sX, _sY):
	Y = _csv.getNumericColumnWithKey(_key)

	#s
	M = np.zeros((_sY, _sX))
	for numTrees in range(_sX):
		for depth in range(_sY):
			y = Y[numTrees*_sY + depth]
			if y==-1:
				y = None

			M[depth][numTrees] = y
	return M


def plotSweetSpot(_file, _sX, _sY):
	csv = CSV()
	csv.load(_file)

	#
	fig, axs = plt.subplots(2,2)
	grid(getMatrix(csv, "r2", _sX, _sY), axs[0][0], "R2")
	grid(getMatrix(csv, "msp", _sX, _sY)/1000, axs[0][1], "MSP430 Program Memory Occupation [kB]")
	grid(getMatrix(csv, "arduino", _sX, _sY)/1000, axs[1][0], "Atmega Program Memory Occupation [kB]")
	grid(getMatrix(csv, "esp", _sX, _sY)/1000, axs[1][1], "ESP32 Program Memory Occupation [kB]")

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




regressionRF()
plotSweetSpot("tmp/rf_regression_mem_.csv", 30, 15)