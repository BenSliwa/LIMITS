import numpy as np
import matplotlib.pyplot as plt
from data.FileHandler import FileHandler
from data.CSV import CSV

class PlotTool:
	def __init__(self):
		""

	def box(self):

		data = [];
		for i in range(0, 2):

			csv = CSV()
			csv.load("tmp/cv_" +  str(i) + ".csv")

			if i==0:
				data = csv.getNumericData()
			else:
				data = [data,  csv.getNumericData()]


		fig1, ax1 = plt.subplots()
		ax1.boxplot(data)
		plt.show()