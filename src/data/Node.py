class Node:
	def __init__(self, _parent, _depth, _condition, _result):
		self.parent = _parent
		self.leftChild = 0
		self.rightChild = 0
		self.level = _depth
		self.condition = _condition
		self.result = _result.strip(" ")

	def printGraph(self):
		out = ""
		for i in range(0, self.level-1):
			out += "_" 
		out += self.condition + " " + self.result
		print(out)

		if self.leftChild!=0:
			self.leftChild.printGraph(self.level+1)
		if self.rightChild!=0:
			self.rightChild.printGraph(self.level+1)

	def getIndent(self):
		result = ""
		for i in range(0, self.level+1):
			result += "\t"
		return result

	def isLeaf(self):
		if self.leftChild==0 and self.rightChild==0:
			return True
		return False