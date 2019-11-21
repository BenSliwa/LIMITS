import numpy as np
import matplotlib.pyplot as plt
from data.FileHandler import FileHandler
from data.CSV import CSV
from plot.PlotTool import PlotTool

class ResultVisualizer:


	def __init__(self):
		""


	def barchart(self, _file, **kwargs):
		M = CSV(_file).toMatrix()
		Y = np.mean(M.data, axis=0)
		S = np.std(M.data, axis=0)

		pt = PlotTool()
		pt.barchart(Y, S, M.header)
		pt.finalize(kwargs)


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


	def scatter(self, _files, _keyX, _keyY, **kwargs):
		pt = PlotTool(kwargs)
		
		for file in _files:
			csv = CSV(file)
			X = csv.getNumericColumnWithKey(_keyX)
			Y = csv.getNumericColumnWithKey(_keyY)

			plt.scatter(X, Y, marker="*", c="blue")
		pt.finalize(kwargs)


	def colormaps(self, _rows, _columns, _files, _titles, **kwargs):
		fig, axs = plt.subplots(_rows, _columns)
		args = kwargs
		args["fig"] = fig

		for row in range(_rows):
			for col in range(_columns):
				index = row * _columns + col
				if index<len(_files):
					args["show"] = False
					if index==len(_files)-1:
						args["show"]  = True

					if _rows==1 or _columns==1:
						c = index
					else:
						c = row,col
					args["ax"] = ax=axs[c]
					args["title"] = _titles[index]
					
					self.colorMap(_files[index], **args)


	def colorMap(self, _file, **kwargs):
		csv = CSV(_file)
		M = csv.getNumericData()
		center = (np.min(M)+np.max(M))/2


		pt = PlotTool(kwargs)
		im = pt.ax.imshow(M, cmap=kwargs.get("cmap", "coolwarm"))

		for i in range(len(M)):
		    for j in range(len(M)):
		    	v = M[i, j]	
		    	color = "k"
		    	if v>center:
		    		color = "w"

		    	text = pt.ax.text(j, i, format(v, '.2f'), ha="center", va="center", color=color)

		pt.ax.set_xticks(np.arange(len(M)))
		pt.ax.set_yticks(np.arange(len(M)))
		pt.ax.set_xticklabels(csv.header)
		pt.ax.set_yticklabels(csv.header)
		pt.ax.set_title(kwargs.get("title", ""))
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