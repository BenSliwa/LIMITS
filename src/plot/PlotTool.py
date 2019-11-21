from data.CSV import CSV
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



class PlotTool:
	def __init__(self, kwargs={}):
		if 'ax' in kwargs:
			self.ax = kwargs.get('ax')
			self.fig = kwargs.get('fig')
		else:
			self.fig, self.ax = plt.subplots()


	def boxplot(self, _data, _labels):
		means = {"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"}
		fliers = {"markerfacecolor":"red", "markeredgecolor":"red"}
		whiskers = {"linestyle": "--"}

		bp = self.ax.boxplot(_data, patch_artist=True, showmeans=True, sym="+", meanprops=means, flierprops=fliers, whiskerprops=whiskers)

		plt.setp(bp["medians"], color="red")
		plt.setp(bp["means"], color="black")
		plt.setp(bp["boxes"], color="blue")

		for patch in bp["boxes"]:
			patch.set(facecolor="white")	


		self.ax.set_xticklabels(_labels)
		

	def barchart(self, _data, _std, _labels):
		self.ax.bar(range(len(_data)), _data, width = 0.5, color = 'blue', edgecolor = 'black', yerr=_std, capsize=7)
		self.ax.set_xticks(range(len(_labels)))
		self.ax.set_xticklabels(_labels)


	def finalize(self, kwargs={}):
		self.ax.set_xlabel(kwargs.get('xlabel', ""))
		self.ax.set_ylabel(kwargs.get('ylabel', ""))

		self.fig.tight_layout()

		if 'xticks' in kwargs:
			xTicks = kwargs.get('xticks')
			self.ax.set_xticks(range(len(xTicks)))
			self.ax.set_xticklabels(xTicks)

		#
		if 'savePNG' in kwargs:
			plt.savefig(kwargs.get('savePNG'), format='png')
		if 'saveEPS' in kwargs:
			plt.savefig(kwargs.get('saveEPS'), format='eps')

		#
		if kwargs.get('show', True):
			plt.show()




	def colorMap(self, _data, _title):
		_data = np.flipud(_data)
		im = self.ax.imshow(_data, cmap="jet")

		divider = make_axes_locatable(self.ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		clb = plt.colorbar(im, cax=cax)

		self.ax.set_title(_title)