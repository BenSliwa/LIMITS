from models.Model import Model
from models.ann.Config import Config
from data.FileHandler import FileHandler
from data.EpsDocument import EpsDocument
from data.CSV import CSV
from code.Compiler import Compiler
from code.CodeGenerator import CodeGenerator
from experiment.Type import Type
import numpy as np

class M5(Model):
	def __init__(self):
		super().__init__("M5")
		self.config = Config()


