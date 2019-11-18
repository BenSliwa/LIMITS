import subprocess
import os
from data.CSV import CSV
from data.FileHandler import FileHandler
from data.ResultMatrix import ResultMatrix
from code.CodeGenerator import CodeGenerator
from code.Compiler import Compiler
from experiment.ConfusionMatrix import ConfusionMatrix
from experiment.Experiment import Type
import numpy as np


class CodeEvaluator:
	def __init__(self):
		self.tempCodeFile = "CodeEvaluation.c"
		self.tempExecutable = "CodeEvaluation.exe"
		self.discretization = None
		self.modelType = Type.CLASSIFICATION


	def execute(self, _cmd):
		result = ""
		try:
			result = subprocess.check_output(_cmd, shell=True).decode("utf-8") 
		except subprocess.CalledProcessError as e:
			result = "invalid"
		return result.rstrip()


	def build(self, _codeFile, _test):
		code = "#include <stdio.h>\n#include <stdlib.h>\n#include <sstream>\n#include <iostream>\n\n" + "\n".join(FileHandler().read(_codeFile)) + "\n" 

		code += "\nint main(int _argc, char* argv[])\n{\n"
		lines = FileHandler().read(_test)
		code += "\tstd::stringstream stream;\n"
		for i in range(1, len(lines)):
			line = lines[i].split(",")

			code += "\tstream << "
			code += self.buildEmbeddedPredictionCall(line[1:])
			
			if i<len(lines)-1:
				code += " << \",\"" 
			code += ";\n"

		code += "\n\tstd::cout << stream.str() << std::endl;\n\n"
		code += "\treturn 0;\n"
		code += "}"

		FileHandler().write(code, self.tempCodeFile)
		Compiler().run(self.tempCodeFile, self.tempExecutable)



	def evaluate(self, _codeFile, _attributes, _test):
		if not "{" in _attributes[0].type:
			return self.regression(_codeFile, _attributes, _test)
		else:
			return self.classification(_codeFile, _attributes, _test)


	def crossValidation(self, _model, _training, _attributes, _discretization=None, **kwargs):
		folds = kwargs.get('xlabel', 10)
		self.discretization = _discretization
		if _attributes[0].type=="NUMERIC":
			self.modelType=Type.REGRESSION		
		else:
			self.modelType=Type.CLASSIFICATION
			
		R = ResultMatrix()
		C = ConfusionMatrix(_attributes[0].type.strip("{").strip("}").split(","))
		fileId = FileHandler().getFileName(_training).replace(".csv", "")
		
		for i in range(folds):
			foldId = fileId + "_" + str(i) + ".csv"
			training = "tmp/training_" + foldId
			test = "tmp/test_" + foldId

			# export the model code
			codeFile = "tmp/code.cpp"
			CodeGenerator().export(training, _model, "id", codeFile, self.discretization)

			# apply the validation
			if self.modelType==Type.REGRESSION:
				keys, results, conf = self.regression(codeFile, _attributes, test, "tmp/predictions_" + str(i) + ".csv")
				R.add(keys, results)
			elif self.modelType==Type.CLASSIFICATION:
				keys, results, conf = self.classification(codeFile, _attributes, test, "tmp/predictions_" + str(i) + ".csv")
				R.add(keys, results)
				C.merge(conf)

		return R, C


	def handlFunctionArguments(self, _attributes):
		if self.discretization:
			V = []
			for i in range(len(_attributes)):
				key = self.discretization.header[i+1]
				V.append(str(self.discretization.discretize(key, float(_attributes[i]))))
			return V
		else:
			return _attributes


	def buildFunctionCall(self, _attributes):
		return self.tempExecutable + " " + " ".join(self.handlFunctionArguments(_attributes))


	def buildEmbeddedPredictionCall(self, _attributes):
		V = self.handlFunctionArguments(_attributes)
		cmd = "predict(" + ",".join(V) + ")"
		if self.discretization and self.modelType==Type.REGRESSION:
			cmd = "(int)" + cmd

		return cmd


	def classification(self, _codeFile, _attributes, _test, _resultFile=""): # att->train, 
		self.modelType = Type.CLASSIFICATION
		self.build(_codeFile, _test)

		classes = _attributes[0].type.strip("{").strip("}").split(",")
		conf = ConfusionMatrix(classes)
	
		predictions = self.execute(self.tempExecutable).split(",")
		labels = CSV(_test).getColumn(0)
		for i in range(len(predictions)):
			conf.update(predictions[i], labels[i])

		accuracy, precision, recall, f_score = conf.calc()
		return ["accuracy", "precision", "recall", "f_score"], np.array([accuracy, precision, recall, f_score]), conf


	def regression(self, _codeFile, _attributes, _test, _resultFile=""): # att->train, 
		self.modelType = Type.REGRESSION
		self.build(_codeFile, _test)

		L = np.array([])
		P = np.array([])

		predictions = self.execute(self.tempExecutable).split(",")
		labels = CSV(_test).getColumn(0)
		for i in range(len(predictions)):
			prediction = float(predictions[i])
			if self.discretization:
				prediction = self.discretization.dediscretize(self.discretization.header[0], prediction)
			L = np.append(L, float(labels[i]))
			P = np.append(P, prediction)

		mae = self.computeMAE(L, P)
		rmse = self.computeRMSE(L, P)
		r2 = self.computeR2(L, P)


		# 
		if _resultFile:
			raw = ResultMatrix()
			raw.add(["label", "prediction"], L)
			raw.add(["label", "prediction"], P)
			raw.data = raw.data.transpose()
			raw.save(_resultFile)

		return ["r2", "mae", "rmse"], np.array([r2, mae, rmse]), ConfusionMatrix()


	def computeMAE(self, _x0, _x1):
		return np.mean(np.absolute(_x0-_x1))


	def computeRMSE(self, _x0, _x1):
		return np.sqrt(np.mean((_x0-_x1)**2))
		

	def computeR2(self, _x0, _x1):
		return np.corrcoef(_x0, _x1)[1][0]**2
