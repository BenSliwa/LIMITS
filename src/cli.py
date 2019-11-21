#!/usr/bin/env python
from models.ann.ANN import ANN
from models.m5.M5 import M5
from models.randomforest.RandomForest import RandomForest
from models.svm.SVM import SVM
from experiment.CrossValidation import CrossValidation
from experiment.Experiment import Type, Experiment
from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.CSV import CSV
from code.CodeGenerator import CodeGenerator
from weka.Weka import WEKA
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino
from plot.ResultVisualizer import ResultVisualizer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classification", type=str, help='path to training data file')
parser.add_argument("-r", "--regression", type=str, help='path to training data file')
parser.add_argument("-m", "--models", type=str, help='descriptors of machine learning models')
parser.add_argument("-n", "--name", type=str, help='name of the result folder', default="cli")
parser.add_argument("-gc", "--gen_code", type=int, default=0)
parser.add_argument("-p", "--platforms", type=str, default=0, help='deployment platform')
parser.add_argument("-cor", "--correlation", type=str, help='path to training data file')
parser.add_argument("-vis", "--visualize", type=str, help='name of the performance indicator')

args = parser.parse_args()


def initModels(_args, _type): 
	models = []
	M = _args.models.split(",")
	for m in M:
		if m=="rf":
			model = RandomForest()
			models.append(model)
		elif m=="m5":
			model =  M5()
			models.append(model)
		elif m=="ann":
			model = ANN()
			models.append(model)
		elif m=="svm":
			model = SVM()
			model.modelType = _type
			models.append(model)
		else:
			print("[ERROR] Model " + m  + "not found", flush=True)

	return models
	

def exportCode(_args, _resultFolder, _training, _models):
	M = _args.models.split(",")
	for i in range(len(M)):
		model = _models[i]
		CodeGenerator().export(_training, model, _resultFolder + M[i] + ".cpp")


def initExperiment(_args):
	FileHandler().createFolder("results")

	resultFolder = "results/" + args.name + "/"
	FileHandler().createFolder(resultFolder)
	resultFile = resultFolder + "result.csv"	

	if _args.classification:
		e = Experiment(args.classification, args.name)
		models = initModels(_args, Type.CLASSIFICATION)
		e.classification(models, 10)

		if _args.gen_code:
			exportCode(_args, resultFolder, _args.classification, models)

		if _args.visualize:
			files = [e.path("cv_" + str(i) + ".csv") for i in range(len(models))] 
			xTicks = [model.modelName for model in models]
			ResultVisualizer().boxplots(files, _args.visualize, xTicks,  ylabel=_args.visualize)

	elif _args.correlation:
		csv = CSV()
		csv.load(args.correlation)
		csv.computeCorrelationMatrix(resultFile)

		if _args.visualize:
			ResultVisualizer().colorMap(resultFile)

	elif _args.regression:
		e = Experiment(args.regression, args.name)
		models = initModels(_args, Type.REGRESSION)
		e.regression(models, 10)

		if _args.gen_code:
			exportCode(_args, resultFolder, _args.regression, models)

		if _args.visualize:
			files = [e.path("cv_" + str(i) + ".csv") for i in range(len(models))] 
			xTicks = [model.modelName for model in models]
			ResultVisualizer().boxplots(files, _args.visualize, xTicks,  ylabel=_args.visualize)

	print("[LIMITS]: results written to src/" + resultFolder)


initExperiment(args)





