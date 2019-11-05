#!/usr/bin/env python
from weka.models.M5 import M5
from weka.models.SVM import SVM
from weka.models.RandomForest import RandomForest
from weka.models.ANN import ANN
from experiment.CrossValidation import CrossValidation
from experiment.Experiment import Type, Experiment
from data.EpsDocument import EpsDocument
from data.FileHandler import FileHandler
from data.CSV import CSV
from code.Forest_Model import Forest_Model
from code.CodeGenerator import CodeGenerator
from weka.Weka import WEKA
from code.MSP430 import MSP430
from code.ESP32 import ESP32
from code.Arduino import Arduino
import numpy as np
import argparse


rf = RandomForest()
rf.trees = 5
rf.printTree = True
rf.depth = 6

# numpy, matplotlib, tkinter



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classification", type=str, help='path to training data file')
parser.add_argument("-r", "--regression", type=str, help='path to training data file')
parser.add_argument("-m", "--models", type=str, help='descriptors of machine learning models')
parser.add_argument("-n", "--name", type=str, help='name of the result folder', default="cli")
parser.add_argument("-gc", "--gen_code", type=int, default=0)
parser.add_argument("-p", "--platforms", type=str, default=0, help='deployment platform')
parser.add_argument("-cor", "--correlation", type=str, help='path to training data file')

args = parser.parse_args()



def initModels(_args, _type): # TODO: add ML type (-> SVM  REG)
	models = []
	M = _args.models.split(",")
	for m in M:
		if m=="rf":
			model = RandomForest()
			model.trees = 5
			model.printTree = True
			model.depth = 6
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
	




def initExperiment(_args):
	FileHandler().createFolder("results")
	FileHandler().createFolder("results/" + args.name)

	if _args.classification:
		e = Experiment(args.classification, args.name)
		models = initModels(_args, Type.CLASSIFICATION)
		e.classification(models, 10)

		if _args.gen_code:
			M = _args.models.split(",")
			for i in range(len(M)):
				model = models[i]
				CodeGenerator().export(args.classification, model, M[i],  "results/" + args.name + "/" + M[i] + ".cpp")

	elif _args.correlation:
		csv = CSV()
		csv.load(args.correlation)
		csv.computeCorrelationMatrix("results/" + args.name + "/corr.csv")

	elif _args.regression:
		e = Experiment(args.regression, args.name)
		models = initModels(_args, Type.REGRESSION)
		e.regression(models, 10)

		if _args.gen_code:
			M = _args.models.split(",")
			for i in range(len(M)):
				model = models[i]
				CodeGenerator().export(args.regression, model, M[i],  "results/" + args.name + "/" + M[i] + ".cpp")



	print("results written to results/" + args.name)

initExperiment(args)





