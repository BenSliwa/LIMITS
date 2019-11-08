from weka.models.RandomForest import RandomForest
from experiment.Experiment import Experiment
from data.CSV import CSV
from data.ResultMatrix import ResultMatrix
from code.CodeGenerator import CodeGenerator
from code.Forest_Model import Forest_Model
from data.FileHandler import FileHandler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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



plotSweetSpot("tmp/rf_regression_mem_.csv", 30, 15)

