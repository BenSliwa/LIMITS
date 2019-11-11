from experiment.Experiment import Type
import abc

class LearningModel(abc.ABC):
	def __init__(self):
		self.modelType = Type.REGRESSION


	def extractLines(self, _data, _start, _end):
		lines = _data.split(_start)[1].split(_end)[0].split("\n")
		lines = [line for line in lines if line.strip() != '']

		return lines


	def extractClasses(self, _attributes):
		return _attributes[0].type.strip("{").strip("}").split(",")


	@abc.abstractmethod
	def serialize(self):
		pass


	@abc.abstractmethod
	def parseResults(self, _data, _config, _results):
		pass

