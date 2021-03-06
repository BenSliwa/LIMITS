from data.FileHandler import FileHandler
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
import numpy as np

class SVM_Model:
	def __init__(self):
		self.classes = []
		self.features = []
		self.offsets = []
		self.weights = []
		self.normedValues = []


	def generateSVMCode(self):
		code = "float svm(float *_values, const float *_weights, float _offset, unsigned int _size)"  + "\n{\n"
		code += "\tfloat sum = 0.0;\n"
		code += CodeGenerator().generateForLoop(1, "unsigned int", "i", 0, "_size")
		code += "\t\tsum += _values[i]*_weights[i];\n"
		code += "\treturn sum + _offset;"
		code += "\n}"

		return code


	def generateClassificationCode(self, _attributes, _classes):
		code = "" 
		code += CodeGenerator().generateArray("const char*", "classes", ["\"" + x + "\"" for x in _classes]) + "\n"	
		
		# compute the weight vectors
		for i in range(0, len(self.weights)):
			w = self.getWeights(self.weights[i], self.features)
			code += CodeGenerator().generateArray("const float", "w" + str(i), w) + "\n"

		code += "\n" + SVM_Model().generateSVMCode() + "\n\n"
		code += CodeGenerator().findMax("int") + "\n\n"
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes)) + "\n{\n"

		# compute the value normalizations
		code += "\t" + CodeGenerator().generateArray("float", "v", self.normedValues) + "\n\n"

 		# one-vs-one
		code += "\t" + CodeGenerator().generateArray("int", "wins", ["0"] * len(_classes)) + "\n"
		for i in range(0, len(self.weights)):	
			c0 = str(_classes.index(self.classes[i][0]))
			c1 = str(_classes.index(self.classes[i][1]))
			code += "\tsvm(v, w" + str(i) + ", " + str(self.offsets[i]) + ", " + str(len(self.features)) + ")<0 ? wins[" + c0 + "]++ : wins[" + c1 + "]++;\n"  
		code += "\n\tunsigned int index = findMax(wins, " + str(len(_classes)) + ");\n\n"

		code += "\treturn classes[index];\n"
		code += "}\n\n"

		return code


	def generateRegressionCode(self, _attributes, _yMin, _yRange):
		code = ""

		# compute the weight vectors
		for i in range(0, len(self.weights)):
			w = self.getWeights(self.weights[i], self.features)
			code += CodeGenerator().generateArray("const float", "w" + str(i), w) + "\n"

		code += "\n" + SVM_Model().generateSVMCode() + "\n\n"
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes)) + "\n{\n"
		code += "\t" + CodeGenerator().generateArray("float", "v", self.normedValues) + "\n\n"
		code += "\tfloat result = svm(v, w0, " + self.offsets[0] + ", " + str(len(self.normedValues)) + ");\n"

		# denormalize the label
		code += "\treturn result * " + str(_yRange) + " " + self.add(_yMin) + ";\n"
		code += "}\n\n"

		return code


	def getWeights(self, _weights, _header):
		w = []
		for i in range(0, len(_header)):
			key = _header[i]
			v  = 0
			if key in _weights:
				v = _weights[key]
			w.append(str(v))
		return w


	def add(self, _x):
		if _x<0:
			return "-" + str(-_x)
		return "+" + str(_x)


	def normalize(self, _csv, _header):
		normedValues = [];
		for j in range(0, len(_header)):
			key = _header[j]
			x = np.array(_csv.getColumn(j+1))
			y = x.astype(np.float)
			r = max(y)-min(y)
			minY = min(y)

			v = ""
			if minY<0:
				v = "(" + key + "+" + str(-minY) + ")/" + str(r)
			else:
				v = "(" + key + "-" + str(minY) + ")/" + str(r)
			normedValues.append(v)

		return normedValues


	def exportWeights(self, _features, _file):
		M = CSV()

		if len(self.classes)>0:
			M.header =  ['class0', 'class1'] + _features
		else:
			M.header = _features


		for c in range(len(self.weights)):
			W = self.weights[c]

			F = [];
			for feature in _features:
				if feature in W:
					F.append(W[feature])
				else:
					F.append(0)

			if len(self.classes)>0:
				M.data.append(','.join(self.classes[c] + [str(x) for x in F]))
			else:
				M.data.append(','.join([str(x) for x in F]))
		M.save(_file)
