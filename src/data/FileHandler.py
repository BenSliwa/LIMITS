import os
import numpy as np


class FileHandler:
	def __init__(self):
		""

	def createFolder(self, _folder):
		try:
			os.mkdir(_folder)
		except FileExistsError:
			""


	def clearFolder(self, _folder):
		""


	def read(self, _file):
		f = open(_file, "r")
		data = f.read()
		f.close()

		return data.splitlines()


	def write(self, _data, _file):
		f = open(_file, "w")
		f.write(_data)
		f.close()


	def append(self, _data, _file):
		f = open(_file, "a")
		f.write(_data)
		f.close()


	def checkFolder(self, _file):
		folder = os.path.dirname(_file)
		if folder:
			self.createFolder(folder)


	def getFileName(self, _path):
		return _path.split("/")[-1]
		

	def saveMatrix(self, _header, _data, _file):
		self.checkFolder(_file)
		np.savetxt(_file, np.asmatrix(_data), delimiter=',', fmt='%f', header=",".join(_header), comments='')


	def saveDict(self, _data, _file):
		header = ""
		values = ""

		keys = list(_data.keys())
		for i in range(0, len(keys)):
			key = keys[i]
			header += key
			values += str(_data[key])

			if i<len(keys)-1:
				header += ","
				values += ","

		self.write(header + "\n" + values, _file)


	def generateId(self, _file):
		id = "0"
		if "/" in _file and "." in _file:
			id = _file.split("/")[-1].split(".")[0]
		return id