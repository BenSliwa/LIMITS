import subprocess
import os
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
from code.CodeGenerator import CodeGenerator
from code.Compiler import Compiler
from experiment.ConfusionMatrix import ConfusionMatrix
import numpy as np


class CodeEvaluator:
	def __init__(self):
		self.tempCodeFile = "CodeEvalutor.c"
		self.tempExecutable = "CodeEvaluation.exe"


	def execute(self, _cmd):
		result = ""
		try:
			result = subprocess.check_output(_cmd, shell=True).decode("utf-8") 
		except subprocess.CalledProcessError as e:
			result = "invalid"
		return result


	def build(self, _codeFile, _attributes, _callType):
		data = "#include <stdio.h>\n#include <stdlib.h>\n\n" + "\n".join(FileHandler().read(_codeFile)) + "\n" + self.generateMain(_attributes, _callType)
		FileHandler().write(data, self.tempCodeFile)
		Compiler().run(self.tempCodeFile, self.tempExecutable)
		

	def evaluate(self, _codeFile, _attributes, _test):
		if not "{" in _attributes[0].type:
			return self.regression(_codeFile, _attributes, _test)
		else:
			return self.classification(_codeFile, _attributes, _test)


	def classification(self, _codeFile, _attributes, _test): # att->train, 
		self.build(_codeFile, _attributes, "const char*")

		classes = _attributes[0].type.strip("{").strip("}").split(",")
		conf = ConfusionMatrix()
		conf.init(classes)
		
		lines = FileHandler().read(_test)
		for i in range(1, len(lines)):
			line = lines[i].split(",")
			conf.update(self.execute(self.tempExecutable + " " + " ".join(line[1:])), line[0])

		accuracy, precision, recall, f_score = conf.calc()
		return ["accuracy", "precision", "recall", "f_score"], np.array([accuracy, precision, recall, f_score]), conf


	def regression(self, _codeFile, _attributes, _test, _resultFile=""): # att->train, 
		self.build(_codeFile, _attributes, "float")

		L = np.array([])
		P = np.array([])
		lines = FileHandler().read(_test)
		for i in range(1, len(lines)):
			line = lines[i].split(",")
			L = np.append(L, float(line[0]))
			P = np.append(P, float(self.execute(self.tempExecutable + " " + " ".join(line[1:]))))

		mae = self.computeMAE(L, P)
		rmse = self.computeRMSE(L, P)
		r2 = self.computeR2(L, P)

		if _resultFile:
			raw = ResultMatrix()
			raw.add(["label", "prediction"], L)
			raw.add(["label", "prediction"], P)
			raw.data = raw.data.transpose()
			raw.save(_resultFile)

		return ["r2", "mae", "rmse"], np.array([r2, mae, rmse]), ConfusionMatrix()


	def generateMain(self, _attributes, _callType):
		numAttributes = len(_attributes)-1

		code = "\nint main(int _argc, char* argv[])\n{\n"
		
		code += "\t" + _callType + " r = predict("
		for i in range(0, numAttributes):
			code += "atof(argv[" + str(i+1) + "])"
			if i<numAttributes-1:
				code += ", "
		code += ");\n"

		if _callType=="const char*":
			code += "\tprintf(r);\n"
		else:
			code += "\tprintf(\"%f\", r);\n"
		code += "\treturn 0;\n"
		code += "}"

		return code


	def computeMAE(self, _x0, _x1):
		return np.mean(np.absolute(_x0-_x1))


	def computeRMSE(self, _x0, _x1):
		return np.sqrt(np.mean((_x0-_x1)**2))
		

	def computeR2(self, _x0, _x1):
		return np.corrcoef(_x0, _x1)[1][0]**2
