from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator
from data.CSV import CSV
from data.FileHandler import FileHandler
from experiment.Type import Type
import numpy as np


class CppSerializer():
	def __init__(self, _model):
		super().__init__()
		self.model = _model


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
		for i in range(0, len(self.model.weights)):
			w = self.model.getWeights(self.model.weights[i], self.model.features)
			code += CodeGenerator().generateArray("const float", "w" + str(i), w) + "\n"

		code += "\n" + self.generateSVMCode() + "\n\n"
		code += CodeGenerator().findMax("int") + "\n\n"
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes)) + "\n{\n"

		# compute the value normalizations
		code += "\t" + CodeGenerator().generateArray("float", "v", self.model.normedValues) + "\n\n"

 		# one-vs-one
		code += "\t" + CodeGenerator().generateArray("int", "wins", ["0"] * len(_classes)) + "\n"
		for i in range(0, len(self.model.weights)):	
			c0 = str(_classes.index(self.model.classes[i][0]))
			c1 = str(_classes.index(self.model.classes[i][1]))
			code += "\tsvm(v, w" + str(i) + ", " + str(self.model.offsets[i]) + ", " + str(len(self.model.features)) + ")<0 ? wins[" + c0 + "]++ : wins[" + c1 + "]++;\n"  
		code += "\n\tunsigned int index = findMax(wins, " + str(len(_classes)) + ");\n\n"

		code += "\treturn classes[index];\n"
		code += "}\n\n"

		return code


	def generateRegressionCode(self, _attributes, _yMin, _yRange):
		code = ""

		# compute the weight vectors
		for i in range(0, len(self.model.weights)):
			w = self.model.getWeights(self.model.weights[i], self.model.features)
			code += CodeGenerator().generateArray("const float", "w" + str(i), w) + "\n"

		code += "\n" + self.generateSVMCode() + "\n\n"
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes)) + "\n{\n"
		code += "\t" + CodeGenerator().generateArray("float", "v", self.model.normedValues) + "\n\n"
		code += "\tfloat result = svm(v, w0, " + self.model.offsets[0] + ", " + str(len(self.model.normedValues)) + ");\n"

		# denormalize the label
		code += "\treturn result * " + str(_yRange) + " " + self.add(_yMin) + ";\n"
		code += "}\n\n"

		return code


	def add(self, _x):
		if _x<0:
			return "-" + str(-_x)
		return "+" + str(_x)
