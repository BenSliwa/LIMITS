import numpy as np
import matplotlib.pyplot as plt
from data.FileHandler import FileHandler
from data.CSV import CSV
from plot.PlotTool import PlotTool

class ResultVisualizer:


	def __init__(self):
		""


	def boxplots(self, _files, _key, _xTickLabels, **kwargs):
		Y = []
		for file in _files:
			csv = CSV(file)
			y = csv.getNumericColumnWithKey(_key)
			Y.append(y)

		pt = PlotTool(kwargs)
		pt.boxplot(Y, _xTickLabels)
		pt.finalize(kwargs)


	def errorbars(self, _files, _key, **kwargs):
		pt = PlotTool(kwargs)

		for file in _files:
			csv = CSV(file)
			y = csv.getNumericColumnWithKey(_key)
			yStd = csv.getNumericColumnWithKey(_key + "_std")
			x = np.arange(1, len(csv.data)+1) * 100

			plt.errorbar(x, y, yerr=yStd, capsize=7)

		pt.finalize(kwargs)



	def scatter(self, _file, _keyX, _keyY, **kwargs):
		pt = PlotTool(kwargs)

		csv = CSV(_file)
		X = csv.getNumericColumnWithKey(_keyX)
		Y = csv.getNumericColumnWithKey(_keyY)

		plt.scatter(X, Y, marker="*")
		pt(kwargs)


	def colorMap(self, _file, **kwargs):
		csv = CSV(_file)
		M = csv.getNumericData()

		pt = PlotTool(kwargs)
		im = pt.ax.imshow(M, cmap="coolwarm")

		for i in range(len(M)):
		    for j in range(len(M)):
		        text = pt.ax.text(j, i, format(M[i, j], '.2f'), ha="center", va="center", color="w")

		pt.ax.set_xticks(np.arange(len(M)))
		pt.ax.set_yticks(np.arange(len(M)))
		pt.ax.set_xticklabels(csv.header)
		pt.ax.set_yticklabels(csv.header)
		plt.setp(pt.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		pt.finalize(kwargs)


	def readAsMatrix(self, _file, _key, _sX, _sY):
		csv = CSV(_file)
		M = np.zeros((_sY, _sX))
		Y = csv.getNumericColumnWithKey(_key)

		for x in range(_sX):
			for y in range(_sY):
				v = Y[x*_sY + y]
				if v==-1:
					v = None

				M[y][x] = v
		return M