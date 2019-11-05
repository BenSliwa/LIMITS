import subprocess
import os
from data.FileHandler import FileHandler
from code.CodeGenerator import CodeGenerator
from settings.Settings import Settings

# Assumes English language setting of the IDE

class Arduino:
	def __init__(self):
		""

	def run(self, _file, _callType, _numAttributes):
		name = _file.split("/")[-1].split(".")[0]
		folder = os.path.dirname(_file) + "/" + name

		FileHandler().createFolder(folder)
		FileHandler().clearFolder(folder)

		file = folder + "/" + name + ".ino"	
		lines = FileHandler().read(_file)
		for i in range(len(lines)):
			line = lines[i]
			if "const float" in line and " = {" in line:
				line = line.replace(" = {", " PROGMEM = {")
				lines[i] = line

		FileHandler().write("\n".join(lines) + self.generateDummyMain(_callType, _numAttributes), file)

		mem = self.compile(file)

		return mem


	def compile(self, _file):
		cmd = Settings().arduinoPath + " --verify " + _file

		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		err = result.stderr.decode()
		out = result.stdout.decode()


		FileHandler().write(out, "tmp/arduino.txt")

		mem = -1
		if "Sketch uses " in out:
			program = float(out.split("Sketch uses ")[1].split(" bytes")[0])
			program_percent = float(out.split("(")[1].split("%)")[0])

			dynamic = float(out.split("Global variables use ")[1].split(" bytes")[0])
			dynamic_percent = float(out.split("(")[2].split("%)")[0])

			if program_percent>100:
				program = -1
			mem = program

		return mem
		

	def generateDummyMain(self, _callType, _numAttributes):
		code = "\n\nvoid setup()\n{\n\tSerial.begin(115200);\n}\n\n"
		code += "void loop()\n{\n"
		code += "\t" + _callType + " r = " + CodeGenerator().generateFunctionCall("predict", ["Serial.read()"]*_numAttributes) + ";\n"
		code += "\tSerial.println(r);\n"
		code += "\n}\n"

		return code


	def toString(self):
		return "Arduino"