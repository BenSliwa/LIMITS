from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.Node import Node
from data.CSV import CSV
from code.Tree_Model import Tree_Model
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator

class Forest_Model:
	def __init__(self):
		self.trees = []
		self.discretization = None


	def exportEps(self, _depth, _numX, _numY, _attributes):
		eps = EpsDocument(2000, 2000)

		l = len(self.trees)
		w = eps.width / _numX

		for i in range(0, len(self.trees)):
			g = self.trees[i]
			root = g.root
			
			x = i%_numX
			y = int(i/_numX)
			xOffset = x * w + w/2
			yOffset = y * eps.height / _numY

			g.drawNode(root, 0, xOffset, eps, 0, 0, w, eps.height - yOffset, _depth, _numY, False)

			if i==3:
				eps2 = EpsDocument(400, 400)
				g.drawNode(root, 0, eps2.width/2, eps2, 0, 0, eps2.width, eps2.height, _depth, 1, True)
				eps2.save("tree.eps")

		eps.save("forest.eps")



	def generateClassificationCode(self, _attributes, _classes):
		code = ""
		classes = ["\"" + x + "\"" for x in _classes]
		code += CodeGenerator().generateArray("const char*", "classes", classes) + "\n\n"	

		# 
		for g in self.trees:
			root = g.root
			treeCode = g.generateGraphCode() + "\n\n"
			for i in range(0, len(classes)):
				key = classes[i]
				treeCode = treeCode.replace("const char* tree", "int tree")
				treeCode = treeCode.replace("return " + key, "return " + str(i))
			code += treeCode

		code += CodeGenerator().findMax("int") + "\n\n"

		# majority decision
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes, self.discretization)) + "\n{\n"
		code += "\t" + CodeGenerator().generateArray("int", "wins", ["0"] * len(_classes)) + "\n"

		for i in range(0, len(self.trees)):
			code += "\twins[" + CodeGenerator().generateFunctionCall("tree_" + str(i), CSV().createAttributeDict(_attributes[1:], self.discretization)) + "]++;\n"

		code += "\tunsigned int index = findMax(wins, " + str(len(_classes)) + ");\n\n"
		code += "\treturn classes[index];\n"
		code += "}"

		return code
		

	def generateRegressionCode(self, _attributes):
		code = ""
		for g in self.trees:
			root = g.root
			code += g.generateGraphCode() + "\n\n"

		# mean
		code += CodeGenerator().generateFunctionHeader("predict", CSV().createAttributeDict(_attributes, self.discretization)) + "\n{\n"

		if self.discretization:
			code += "\tint sum = 0;\n"
		else:
			code += "\tfloat sum = 0;\n"


		for i in range(0, len(self.trees)):
			code += "\tsum += " + CodeGenerator().generateFunctionCall("tree_" + str(i), CSV().createAttributeDict(_attributes[1:], self.discretization)) + ";\n"

		if self.discretization:
			code += "\n\treturn sum / " + str(len(self.trees)) + ";\n"

			# TODO: this would be required to undo the discretization, however we skip it here as we want a fully discretized model - it is assumed to dediscretization is done at the application level
			#code += "\n\treturn (sum / " + str(len(self.trees)) + ") * " + str((self.discretization.widths[0])) + " + " + str((self.discretization.min[0])) + ";\n"
		else:
			code += "\n\treturn sum / " + str(len(self.trees)) + ".0;\n"
		code += "}"

		return code


	def parse(self, _data, _attributes):
		trees = _data.split("RandomTree\n==========")
		l = len(trees)-1
		for i in range(1, l+1):
			items = trees[i].split("Size of the tree")
			tree = items[0].split("\n")
			s = float(items[1].split("\n")[0].strip("Size of the tree : "))

			g = Tree_Model(str(i-1))
			g.discretization = self.discretization

			root = g.init(tree, _attributes)
			self.trees.append(g)

