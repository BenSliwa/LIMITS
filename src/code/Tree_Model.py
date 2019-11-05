from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.CSV import CSV
from data.Node import Node


class Tree_Model:
	def __init__(self, _id):
		self.nodes = 1
		self.leaves = 0
		self.depth = 0
		self.root = 0
		self.attributes = {}
		self.labelType = ""
		self.id = _id
		self.linearModels = []

		self.useDiscretization = False


	def genNodeCode(self, _node):
		result = ""
		indent = _node.getIndent()

		if _node.isLeaf():
			if self.labelType=="const char*":
				result += indent + "return \"" + self.handleDiscretization(_node.result) + "\";\n"
			else:
				result += indent + "return " + self.handleDiscretization(_node.result) + ";\n"
		else:
			result += indent + "if(" + self.handleDiscretization(_node.leftChild.condition) + ")\n"
			if _node.leftChild.isLeaf():
				result += self.genNodeCode(_node.leftChild)
			else:
				result += indent + "{\n" + self.genNodeCode(_node.leftChild) + indent + "}\n"

			result += indent + "else\n"
			if _node.rightChild.isLeaf():
				result += self.genNodeCode(_node.rightChild)
			else:
				result += indent + "{\n" + self.genNodeCode(_node.rightChild) + indent + "}\n"

		return result


	def handleDiscretization(self, _cmd):
		if self.useDiscretization:
			if " < " in _cmd:
				cmd = _cmd.split(" < ")
				v = round(float(cmd[1]))
				return cmd[0] + " < " + str(v)
			else:
				v = round(float(_cmd))
				return str(v)
		return _cmd


	def generateHeader(self, _key):	# TODO: move to CodeGen
		result = ""
		i = 0
		for key in self.attributes.keys():
			dataType = self.attributes[key]
			if i==0:
				self.labelType = dataType
				result += dataType + " " + _key + "("
			else:
				result += dataType + " " + key
				if i<len(self.attributes.keys())-1:
					result += ", "
			i += 1
		result += ")\n{\n"

		return result

	def generateGraphCode(self):
		result = self.generateHeader("tree_" + self.id)
		result += self.genNodeCode(self.root)
		result += "}"

		#
		if len(self.linearModels)>0:
			for key in list(self.linearModels.keys()):
				result = result.replace(key + ";", self.linearModels[key] + ";")

		return result


	def init(self, _lines, _attributes):
		self.attributes = CSV().createAttributeDict(_attributes, self.useDiscretization)

		lines = _lines

		self.root = Node(0, 0, "", "")
		lastNode = self.root
		depth = 0

		for line in lines:
			if len(line)==0:
				continue

			d = len(line.split("|"))
			condition = line.split("|")[-1].split(" : ")[0].strip(" ")

			result = ""
			if " : " in line and "(" in line:
				result = line.split("|")[-1].strip(" ").split(" : ")[1].split("(")[0]
				self.leaves += 1

				# TODO: IF THE RESULT IS NOMIMAL, ADD ""

			self.depth = max(d, self.depth)
			if d>depth: # left child -> if
				node = Node(lastNode, d, condition, result)
				self.nodes += 1

				lastNode.leftChild = node
				lastNode = node
			elif d<depth:
				delta = depth - d
				parent = lastNode.parent
				for i in range(0, delta):
					parent = parent.parent

				node = Node(parent, d, "else", result)
				self.nodes += 1

				parent.rightChild = node
				lastNode = node
			else: # right child -> else
				parent = lastNode.parent
				node = Node(parent, d, "else", result)
				parent.rightChild = node
				lastNode = node
			depth = d
		
		return self.root


	def drawNode(self, _node, _level, _x, _eps, _pX, _py, _width, _height, _depth, _numY, _printCondition):
		yInc = _eps.height / _numY / _depth
		y = _height - _level * yInc

		# 
		_eps.setColor(0, 0, 255)
		if _level>0:
			_eps.drawLine(_pX, _py, _x, y)
			
		if _node.isLeaf():
			_eps.startPath()
			_eps.data += str(_x) + " " + str(y) + " " + str(1) + " 0 360 arc "
			_eps.closePath();
			_eps.setColor(255, 0, 0)
			_eps.fill()
		else:
			_eps.drawCircle(_x, y, 1, "")

		xOffset = _width / 2**(_level+2)

		#
		if _node.rightChild!=0:
			self.drawNode(_node.rightChild, _level+1, _x+xOffset, _eps, _x, y, _width, _height, _depth, _numY, _printCondition)
		if _node.leftChild!=0:
			self.drawNode(_node.leftChild, _level+1, _x-xOffset, _eps, _x, y, _width, _height, _depth, _numY, _printCondition)
			
		# print the text overlay
		_eps.setColor(0, 0, 0)
		if _printCondition:
			if _level>0:
				tX = (_x + _pX) / 2
				tY = (y + _py) / 2
				if _node==_node.parent.leftChild:
					_eps.text(_node.condition, 5, _pX, _py, 0, "-0.5", "-0.5")

			if _node.isLeaf():
				_eps.text(_node.result, 5, _x, y, 0, "-.6", "-1.2")

