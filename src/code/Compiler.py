import subprocess
import os
from data.FileHandler import FileHandler

class Compiler:
	def __init__(self):
		self.cmd = "g++ "

	def run(self, _file, _result):
		cmd = self.cmd + _file + " -o " + _result

		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		err = result.stderr.decode()
		out = result.stdout.decode()

		#if err:
		#	print([err])


	def computeSize(self, _file):
		cmd = "size " + _file
		out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
		lines = out.split("\n")

		dec = -1
		if len(lines)>1:
			dec = lines[1].split("\t")[3].strip(" ")

		return dec












