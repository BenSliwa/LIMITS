from experiment.Experiment import Type
import abc

class Platform(abc.ABC):
	def __init__(self):
		""


	@abc.abstractmethod
	def run(self, _file, _callType, _numAttributes):
		pass

