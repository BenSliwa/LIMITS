from data.FileHandler import FileHandler

class Configuration:
	def __init__(self, _training, _model, _folds):
		self.training = _training
		header = FileHandler().read(self.training)[0].split(",")
		self.label = header[0]
		self.features = header[1:]

		self.model = _model
		self.folds = _folds
		self.tmpFolder = "tmp/"




