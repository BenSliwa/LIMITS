from data.FileHandler import FileHandler
from data.EpsDocument import EpsDocument
from data.CSV import CSV
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator
from weka.Weka import WEKA
from experiment.Experiment import Type
import numpy as np


class ANN_Model:
	def __init__(self):
		self.training = ""
		self.inputLayer = [] # keys
		self.outputLayer = [] # keys
		self.layers = [] # weight and treshold matrices [hidden + out] 
		self.L = [] # hidden layer network structure
		self.modelType = Type.REGRESSION
		self.useUnrolling = True


	def generateCode(self, _file):
		csv = CSV()
		csv.load(self.training)
		attributes = csv.findAttributes(0)
		normed = self.normalize(csv, attributes)
		resultType = "float"

		code = "#include <math.h>\n"
		if self.modelType==Type.CLASSIFICATION:
			code += ""
			classes = attributes[0].type.strip("{").strip("}").split(",")
			classes = ["\"" + key + "\"" for key in classes]

			code += CodeGenerator().generateArray("const char*", "classes", classes) + "\n\n"
			resultType = "const char*"
		else:
			code += "\n"

		# weight matrices
		if not self.useUnrolling:
			for i in range(0, len(self.layers)):
				W = self.layers[i][0]
				name = "w" + str(i)
				if i==len(self.layers)-1:
					name = "w_out"

				code += "const " + CodeGenerator().generateMatrix("float", name, W) + "\n"
			code += "\n"

		# threshold vectors
		for i in range(0, len(self.layers)):
			matrix = self.layers[i]
			T = self.layers[i][1]
			name = "th" + str(i)
			if i==len(self.layers)-1:
				name = "th_out"

			code += "const " + CodeGenerator().generateArray("float", name, T) + "\n"
		code += "\n"

		# generate the required ann-specific methods
		code += self.sigmoid() + "\n\n"
		code += self.activate() + "\n\n"
		if not self.useUnrolling:
			code += self.mult() + "\n\n"

		if self.modelType==Type.CLASSIFICATION:
			code += CodeGenerator().findMax("float") + "\n\n"

		# generate the callable method
		header = ["_" + key for key in self.inputLayer]
		code += resultType + " predict(" + ", ".join(["float " + x for x in header]) + ")\n{\n"

		# input layer
		for i in range(0, len(header)):
			header[i] = self.norm(header[i], normed[i+1][0], normed[i+1][1])
		code += "\t" + CodeGenerator().generateArray("float", "in", header) + "\n\n"

		# activate the layers
		if self.useUnrolling:
			code += self.activateLayersWithUnrolling(normed)
		else:
			code += self.activateLayers(header, normed)



		code += "}\n"

		#code += CodeGenerator().generateDummyMain(len(attributes)-1)

		FileHandler().write(code, _file)

	def activateLayers(self, _header, _normed):
		code = ""
		lastLayer = "in"
		for i in range(0, len(self.L)):
			layer = self.L[i]

			m = 0
			if i==0:
				m = len(_header)
			else:
				m = len(self.L[i-1])
			n = len(layer)

			code += "\tfloat z" + str(i) + "[" + str(n) + "] = {0};\n"
			code += "\tmult(" + lastLayer + ", &w" + str(i) + "[0][0], z" + str(i) + ", " + str(m) + ", " + str(n) + ");\n"
			code += "\tactivate(z" + str(i) + ", th" + str(i) + ", " + str(n) + ");\n\n"
			lastLayer = "z" + str(i)

		# output layer
		m = len(self.layers[-1][0])
		nOut = str(len(self.outputLayer))
		code += "\tfloat out[" + nOut + "] = {0};\n"
		code += "\tmult(" + lastLayer + ", &w_out[0][0], out, " + str(m) + ", " + nOut + ");\n"

		if self.modelType==Type.CLASSIFICATION:
			code += "\tactivate(out, th_out, " + nOut + ");\n\n"
			code += "\tunsigned int index = findMax(out, " + nOut + ");\n\n"
			code += "\treturn classes[index];\n"
		else:
			code += "\n\treturn (out[0] + 1 + th_out[0]) * " + CodeGenerator().float2String(_normed[0][1]) + " / 2.0 + " + CodeGenerator().float2String(_normed[0][0]) + ";\n"
		return code


	def activateLayersWithUnrolling(self, _normed):
		code = ""
		for i in range(len(self.layers)):
			vIn = "z" + str(i-1)
			vOut = "z" + str(i)
			if i==0:
				vIn = "in"
			if i==len(self.layers)-1:
				vOut = "out"

			code += CodeGenerator().unrollMultiplication(vOut, vIn, self.layers[i][0])
			if i<len(self.layers)-1:
				code += "\tactivate(" + vOut + ", th" + str(i) + ", " + str(len(self.layers[i][0][0])) + ");\n"

		nOut = str(len(self.outputLayer))
		if self.modelType==Type.CLASSIFICATION:
			code += "\tactivate(out, th_out, " + nOut + ");\n\n"
			code += "\tunsigned int index = findMax(out, " + nOut + ");\n"
			code += "\treturn classes[index];\n"
		else:
			code += "\n\treturn (out[0] + 1 + th_out[0]) * " + CodeGenerator().float2String(_normed[0][1]) + " / 2 + " + CodeGenerator().float2String(_normed[0][0]) + ";\n"

			code += "\n"
		return code


	def normalize(self, _csv, _attributes):
		normedValues = [];
		for i in range(0, len(_csv.header)):
			key = _csv.header[i]

			if _attributes[i].type=="NUMERIC":
				x = np.array(_csv.getColumn(i))
				y = x.astype(np.float)
				yMin = min(y)
				r = max(y)-yMin

				normedValues.append([yMin, r])
			else:
				normedValues.append([0, 1])

		return normedValues


	def sigmoid(self):
		code = "float sigmoid(float _x)\n{\n"
		code += "\treturn 1.0 / (1.0 + exp(-_x));\n"
		code += "}"

		return code


	def activate(self):
		code = "void activate(float *_values, const float *_thresholds, unsigned int _size)\n{\n"
		code += CodeGenerator().generateForLoop(1, "unsigned int", "i", 0, "_size")
		code += "\t\t_values[i] = sigmoid(_values[i] + _thresholds[i]);\n"
		code += "}"

		return code


	def mult(self):
		code = "void mult(float *_in, const float *_matrix, float *_out, unsigned int _m, unsigned int _n)\n{\n"

		code += CodeGenerator().generateForLoop(1, "unsigned int", "x", 0, "_n") + "\t{\n"
		code += "\t\t_out[x] = 0;\n"
		code += CodeGenerator().generateForLoop(2, "unsigned int", "y", 0, "_m")

		code += "\t\t\t_out[x] += _in[y] * _matrix[x + y * _n];\n"
		code += "\t}\n"
		code += "}"

		return code


	def norm(self, x, xMin, xRange):
		code = ""
		if xMin<0:
			code = "2*(" + x + "+" + str(-xMin) + ")/" + str(xRange) + "-1"
		else:
			code = "2*(" + x + "-" + str(xMin) + ")/" + str(xRange) + "-1"

		return code


	def exportEps(self, _file):
		eps = EpsDocument(900, 600)

		layers = [len(self.inputLayer)];
		for l in self.layers:
			layers.append(len(l[1]))
		#layers.append(len(self.outputLayer))

		w = eps.width / len(layers)
		lastLayer = []
		h = 40 # eps height / max layer size
		for i in range(0, len(layers)):
			l = layers[i]
			x = i*w + 50
			yOffset = (eps.height - h*l)/2 
			layer = []
			for j in range(0, l): 
				y = yOffset + j * h
				y += h/2

				eps.drawCircle(x, y, 5, "")
				layer.append([x, y])

				if i==0: # inputLayer
					eps.text(self.inputLayer[j], 12, x - w/10, y, 0, "-1", "-0.5")
				elif i==len(layers)-1: # output layer
					eps.text(self.outputLayer[j], 12, x + w/10, y, 0, "0", "-0.5")

				for p in lastLayer:
					eps.drawLine(x, y, p[0], p[1])
			lastLayer = layer

		eps.save(_file)