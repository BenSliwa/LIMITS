import subprocess
from data.FileHandler import FileHandler
from settings.Settings import Settings

class WEKA:
	def __init__(self):
		self.predictions = False 

	def run(self, _cmd, _id):
		result = subprocess.run(_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		err = result.stderr.decode()
		out = result.stdout.decode()

		FileHandler().write(out, "tmp/raw" + _id + ".txt")
		FileHandler().write(err, "tmp/err" + _id + ".txt")
		return out

	def applyModel(self, _model, _training, _test, _id):
		cmd = "java -cp " + Settings().wekaPath + " -Xmx512m "
		cmd += _model.serialize()
		cmd += " -c 1 -t " + _training + " -T " + _test

		if self.predictions:
			cmd += " -classifications weka.classifiers.evaluation.output.prediction.CSV"

		return self.run(cmd, _id)

	def train(self, _model, _training, _id):
		cmd = "java -cp " + Settings().wekaPath + " -Xmx512m "
		cmd += _model.serialize()
		cmd += " -c 1 -t " + _training + " -no-cv"

		return self.run(cmd, _id)
