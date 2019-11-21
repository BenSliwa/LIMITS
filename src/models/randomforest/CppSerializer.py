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
		

	def generateClassificationCode(self, _attributes, _classes):
		code = ""
		classes = ["\"" + x + "\"" for x in _classes]
		code += CodeGenerator().generateArray("const char*", "classes", classes) + "\n\n"	

		# 
		for g in self.model.trees:
			root = g.root
			treeCode = g.generateGraphCode() + "\n\n"
			for i in range(0, len(classes)):
				key = classes[i]
				treeCode = treeCode.replace("const char* tree", "int tree")
				treeCode = treeCode.replace("return " + key, "return " + str(i))
			code += treeCode

		code += CodeGenerator().findMax("int") + "\n\n"

		# majority decision
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes, self.model.discretization)) + "\n{\n"
		code += "\t" + CodeGenerator().generateArray("int", "wins", ["0"] * len(_classes)) + "\n"

		for i in range(0, len(self.model.trees)):
			code += "\twins[" + CodeGenerator().generateFunctionCall("tree_" + str(i), CSV().createAttributeDict(_attributes[1:], self.model.discretization)) + "]++;\n"

		code += "\tunsigned int index = findMax(wins, " + str(len(_classes)) + ");\n\n"
		code += "\treturn classes[index];\n"
		code += "}"

		return code
		

	def generateRegressionCode(self, _attributes):
		code = ""
		for g in self.model.trees:
			root = g.root
			code += g.generateGraphCode() + "\n\n"

		# mean
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes, self.model.discretization)) + "\n{\n"

		if self.model.discretization:
			code += "\tint sum = 0;\n"
		else:
			code += "\tfloat sum = 0;\n"


		for i in range(0, len(self.model.trees)):
			code += "\tsum += " + CodeGenerator().generateFunctionCall("tree_" + str(i), CSV().createAttributeDict(_attributes[1:], self.model.discretization)) + ";\n"

		if self.model.discretization:
			code += "\n\treturn sum / " + str(len(self.model.trees)) + ";\n"

			# TODO: this would be required to undo the discretization, however we skip it here as we want a fully discretized model - it is assumed to dediscretization is done at the application level
			#code += "\n\treturn (sum / " + str(len(self.model.trees)) + ") * " + str((self.model.discretization.widths[0])) + " + " + str((self.model.discretization.min[0])) + ";\n"
		else:
			code += "\n\treturn sum / " + str(len(self.model.trees)) + ".0;\n"
		code += "}"

		return code