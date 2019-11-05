from data.FileHandler import FileHandler

class Attribute:
	def __init__(self):
		self.name = "";
		self.type = "";

class ARFF:
	def __init__(self, _id="0"):
		self.id = _id;

	def serialize(self, _attributes, _data):
		data = "@RELATION " + self.id + "\n\n"
		for i in range(0, len(_attributes)):
			att = _attributes[i]
			data += "@ATTRIBUTE " + att.name + " " + att.type + "\n"
		data += "\n@DATA\n"
		data += "\n".join(_data)

		return data

