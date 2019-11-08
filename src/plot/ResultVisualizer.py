import numpy as np
import matplotlib.pyplot as plt
from data.FileHandler import FileHandler
from data.CSV import CSV


class ResultVisualizer:
	def __init__(self):
		""


	def errorbars(self, _files, _key, **kwargs):
		for file in _files:
			csv = CSV()
			csv.load(file)

			y = csv.getNumericColumnWithKey(_key)
			yStd = csv.getNumericColumnWithKey(_key + "_std")
			x = np.arange(1, len(csv.data)+1) * 100

			plt.errorbar(x, y, yerr=yStd, capsize=7)

		self.plotSetup(kwargs)



	def plotSetup(self, kwargs):
		plt.xlabel(kwargs.get('xlabel', ""))
		plt.ylabel(kwargs.get('ylabel', ""))

		fig = plt.gcf()
		fig.tight_layout()

		if 'xticks' in kwargs:
			xTicks = kwargs.get('xticks')
			plt.xticks([r for r in range(len(xTicks))], xTicks)

		#
		if 'savePNG' in kwargs:
			plt.savefig(kwargs.get('savePNG'), format='png')
		if 'saveEPS' in kwargs:
			plt.savefig(kwargs.get('saveEPS'), format='eps')

		#
		plt.show()


	def scatter(self, _file, _keyX, _keyY, **kwargs):
		csv = CSV()
		csv.load(_file)

		X = csv.getNumericColumnWithKey(_keyX)
		Y = csv.getNumericColumnWithKey(_keyY)

		plt.scatter(X, Y, marker="*")
		self.plotSetup(kwargs)


	def barChart(self, _file, _key, _xTicks, **kwargs):
		csv = CSV()
		csv.load(_file)

		y = csv.getNumericColumnWithKey(_key)	
		yStd = csv.getNumericColumnWithKey(_key + "_std")	
		x = np.arange(len(y))

		plt.bar(x, y, width = 0.5, color = 'blue', edgecolor = 'black', yerr=yStd, capsize=7)

		kwargs["xticks"] = _xTicks
		self.plotSetup(kwargs)


	def colorMap(self, _file, **kwargs):
		csv = CSV()
		csv.load(_file)
		M = csv.getNumericData()


		fig, ax = plt.subplots()
		im = ax.imshow(M, cmap="coolwarm")

		for i in range(len(M)):
		    for j in range(len(M)):
		        text = ax.text(j, i, format(M[i, j], '.2f'), ha="center", va="center", color="w")

		ax.set_xticks(np.arange(len(M)))
		ax.set_yticks(np.arange(len(M)))
		ax.set_xticklabels(csv.header)
		ax.set_yticklabels(csv.header)
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		self.plotSetup(kwargs)