import numpy as np
import math
import random
from data.FileHandler import FileHandler
from data.ARFF import ARFF, Attribute
from data.ResultMatrix import ResultMatrix
from data.Discretization import Discretization

class CSV:
	def __init__(self, _file=""):
		self.header = []
		self.data = []
		self.id = "0"
		self.file = _file

		if self.file:
			self.load(self.file)


	def load(self, _file):
		self.data = FileHandler().read(_file)
		self.header = self.data[0].split(",")
		self.data = self.data[1:]
		self.file = _file
		self.id = FileHandler().generateId(_file)


	def save(self, _file):
		FileHandler().write(",".join(self.header) + "\n" + "\n".join(self.data), _file)


	def removeColumnWithKey(self, _key):
		index = self.header.index(_key)
		if index>-1:
			del self.header[index]
			for i in range(len(self.data)):
				line = self.data[i].split(",")
				del line[index]
				self.data[i] = ",".join(line)


	def randomize(self, _seed):
		self.addIndices()
		random.Random(_seed).shuffle(self.data)


	def stratify(self, _folds, _folder):
		labels = self.getColumn(1)
		data = [];
		for i in range(0, _folds):
			csv = CSV()
			csv.header = self.header
			data.append(csv)

		classes = self.findNominalClasses(labels)
		keys = list(classes.keys())

		for key in keys:
			lines = []
			indices = classes[key]
			for index in indices:
				lines.append(self.getRow(index))

			foldSize = math.ceil(len(lines) / _folds)
			for i in range(0, _folds):
				s = min(len(lines), foldSize)
				data[i].data += lines[0:s]
				lines = lines[s:]

		self.exportFoldData(data, _folder)


	def addIndices(self):
		for i in range(0, len(self.data)):
			self.data[i] = str(i) + "," + self.data[i] 

		
	def createFolds(self, _folds, _folder):
		foldSize = math.ceil(len(self.data) / _folds)
		data = self.data
		folds = [];
		while len(data)>0:
			csv = CSV()
			csv.header = self.header
			s = min(len(data), foldSize)
			csv.data = data[0:s]
			data = data[s:]
			folds.append(csv)
		self.exportFoldData(folds, _folder)


	def exportFoldData(self, _folds, _folder):
		fh = FileHandler()
		folds = len(_folds)

		# generate the index mapping
		indices = [];
		for i in range(0, folds):
			subset = _folds[i]
			subsetIndices = subset.removeIndices()
			indices += subsetIndices
		fh.write("\n".join(indices), _folder + "indices_" + self.id + ".csv")

		# generate the fold data
		for i in range(0, folds):
			train = _folder + "training_" + self.id + "_" + str(i) + ".csv"
			test = _folder + "test_" + self.id + "_" + str(i) + ".csv"

			fh.write(",".join(self.header) + "\n", train)
			fh.write(",".join(self.header) + "\n", test)

			for j in range(0, folds):
				subset = _folds[j]
			
				if i!=j:
					fh.append("\n".join(subset.data) + "\n", train)
				else:
					fh.append("\n".join(subset.data) + "\n", test)

		# generate ARFF files
		arff = ARFF(self.id)
		attributes = self.findAttributes(1)
		for i in range(0, folds):
			train = CSV();
			train.load(_folder + "training_" + self.id + "_" + str(i) + ".csv")			
			test = CSV();
			test.load(_folder + "test_" + self.id + "_" + str(i) + ".csv")

			fh.write(arff.serialize(attributes, train.data), _folder + "training_" + self.id + "_" + str(i) + ".arff")
			fh.write(arff.serialize(attributes, test.data), _folder + "test_" + self.id + "_" + str(i) + ".arff")


	def convertToARFF(self, _file, _removeIndices=True):		
		if _removeIndices:
			self.removeIndices()
		attributes = self.findAttributes(0)
		FileHandler().write(ARFF(self.id).serialize(attributes, self.data), _file)


	def removeIndices(self):
		indices = []
		for i in range(0, len(self.data)):
			line  = self.data[i].split(",")
			indices.append(line[0])
			self.data[i] = ",".join(line[1:])
		return indices


	def getRow(self, _index):
		if _index < len(self.data):
			return self.data[_index]
		return []


	def getColumn(self, _index):
		column = [];
		for line in self.data:
			column.append(line.split(",")[_index])
		return column


	def getNumericColumnWithKey(self, _key):
		index = self.header.index(_key)
		if index>-1:
			return np.array(self.getColumn(index)).astype("float")		
		return np.array([]);		


	def getNumericData(self):
		M = np.array([])
		for y in range(len(self.data)):
			x = np.array(self.data[y].split(",")).astype(np.float)

			if np.prod(M.shape)==0:
				M = x
			else:
				M = np.vstack([M, x])

		return M


	def findAttributes(self, _offset):
		attributes = [];
		columns = len(self.header)
		
		for i in range(0, columns):
			key = self.header[i]
			column = self.getColumn(i+_offset) # skip the first column, which contains the index values
			item = column[0]
			numeric = item.replace('.','',1).replace("-","",1).isdigit()

			att = Attribute()
			att.name = key
			if numeric==True:
				att.type = "NUMERIC"
			else:
				classes = self.findNominalClasses(column)
				att.type = "{" + ",".join(list(classes.keys())) + "}"
			attributes.append(att)
		return attributes


	def createAttributeDict(self, _attributes, _useDiscretization=None):
		attributes = {}
		for att in _attributes:
			dataType = att.type
			if dataType!="NUMERIC":
				dataType = "const char*"
			else:
				if _useDiscretization:
					dataType = "unsigned char"
				else:
					dataType = "float"

			attributes[att.name] = dataType

		return attributes

		
	def findNominalClasses(self, _column):
		classes = {}
		for i in range(0, len(_column)):
			key = _column[i];

			if key in classes:
				classes[key].append(i)
			else:
				classes[key] = [i]
		return classes


	def discretizeData(self):
		d = Discretization()
		attributes = self.findAttributes(0)
		for i in range(len(attributes)):
			attribute = attributes[i]

			d.header.append(self.header[i])
			if attribute.type=="NUMERIC":
				inMin, inMax, binWidth = self.discretizeColumn(i)
				d.min.append(inMin)
				d.max.append(inMax)
				d.widths.append(binWidth)
			else:
				d.min.append(0)
				d.max.append(0)
				d.widths.append(0)

		return d


	def discretizeColumn(self, _index, _range=256):
		data, inMin, inMax, binWidth = self.discretize([float(i) for i in self.getColumn(_index)], _range)
		col = [str(i) for i in data]

		for i in range(0, len(self.data)):
			line = self.data[i].split(",")
			line[_index] = col[i];
			self.data[i] = ",".join(line)
			
		return inMin, inMax, binWidth


	def discretize(self, _data, _range=256): # _data = float array
		inMin = min(_data)
		inMax = max(_data)
		bins = _range
		binWidth = (inMax-inMin) / (bins-1)

		data = []
		for x in _data:
			binId = math.floor((x-inMin)/binWidth)
			data.append(binId)

		return data, inMin, inMax, binWidth


	def pearson(self, _v0, _v1):
		return np.corrcoef(_v0, _v1)[0][1]


	def computeCorrelationMatrix(self, _out):
		M = ResultMatrix()
		s = len(self.data[0].split(","));
		for y in range(s):
			m = [];
			for x in range(s):
				v0 = [float(i) for i in self.getColumn(x)]
				v1 = [float(i) for i in self.getColumn(y)]

				m.append(self.pearson(v0, v1))
			M.add(self.header, np.array(m))

		M.save(_out)


	def toMatrix(self):
		M = ResultMatrix()
		M.header = self.header
		M.data = self.getNumericData()

		return M
		