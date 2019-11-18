import subprocess
import os
from code.Platform import Platform
from data.FileHandler import FileHandler
from code.CodeGenerator import CodeGenerator
from code.Compiler import Compiler
from settings.Settings import Settings

# https://github.com/espressif/esp-idf/tree/master/examples/peripherals/uart/uart_echo_rs485
# https://docs.espressif.com/projects/esp-idf/en/latest/api-reference/peripherals/uart.html
# idf.py -p COM14 flash
# idf.py -p COM14 monitor

class ESP32(Platform):
	def __init__(self):
		super().__init__()


	def run(self, _file, _callType, _numAttributes):
		code = "#include <stdio.h>\n#include <stdlib.h>\n\n"
		code += "\n".join(FileHandler().read(_file))
		code += self.generateDummyMain(_callType, _numAttributes)

		# 
		file = _file.split(".")[0] + ".esp32"
		FileHandler().write(code, file)

		#
		FileHandler().write(code, Settings().espProject + "main/hello_world_main.c")

		return self.compile()


	def compile(self):
		cmd = Settings().espPath + " -C " + Settings().espProject + " build size"

		mem = -1
		try:
			result = str(subprocess.check_output(cmd, shell=True))
			if "Total image size:~" in result:
				mem = float(self.split(result, "Total image size:~", " bytes"))
		except subprocess.CalledProcessError as e:
			mem = -1

		return mem


	def generateDummyMain(self, _callType, _numAttributes):
		code = "\nint app_main(int _argc, char* argv[])\n{\n"	
		code += "\t" + _callType + " r = predict("
		for i in range(0, _numAttributes):
			code += "atof(argv[" + str(i+1) + "])"
			if i<_numAttributes-1:
				code += ", "
		code += ");\n"

		if _callType=="const char*" or _callType=="unsigned char":
			code += "\tprintf(r);\n"
		else:
			code += "\tprintf(\"%f\", r);\n"
		code += "\treturn 0;\n"
		code += "}"

		return code


	def split(self, _data, _start, _end):
		data = ""
		if _start in _data:
			data = _data.split(_start)[1].split(_end)[0]

		return data


	def toString(self):
		return "ESP32"