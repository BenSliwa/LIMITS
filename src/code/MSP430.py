import subprocess
import os
from data.FileHandler import FileHandler
from code.CodeGenerator import CodeGenerator
from code.Compiler import Compiler
from settings.Settings import Settings

class MSP430:
	def __init__(self):
		""

	def run(self, _file, _name, _callType, _numAttributes):
		file = _file.split(".")[0] + ".msp430"
		code = "#include <msp430.h>\n\n" + "\n".join(FileHandler().read(_file)) + "\n\n"
		code += "void printf(char *, ...);\n"
		code += self.generateDummyMain(_callType, _numAttributes)

		FileHandler().write(code, file)

		self.compile(file)
		result = self.link(file, _name)
		return float(Compiler().computeSize(result))
		

	def compile(self, _file):
		cmd = Settings().mspPath + "/tools/compiler/ti-cgt-msp430_18.12.2.LTS/bin/cl430 -vmsp -Ooff --opt_for_speed=0 --use_hw_mpy=none" 
		cmd += " --include_path=" + Settings().mspPath + "/ccs_base/msp430/include" 
		cmd += " --include_path=" + Settings().mspPath + "/tools/compiler/ti-cgt-msp430_18.12.2.LTS/include" 
		cmd += " --advice:power=all --define=__MSP430G2553__ -g --printf_support=minimal --diag_warning=225" 
		cmd += " --diag_wrap=off --display_error_number --preproc_with_compile " + _file
		cmd += " printf.c"

		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		err = result.stderr.decode()
		out = result.stdout.decode()

		#print([out, err])


	def link(self, _file, _name):
		file = _file.split(".")[0] + ".obj"
		cmd = Settings().mspPath + "/tools/compiler/ti-cgt-msp430_18.12.2.LTS/bin/cl430 -vmsp -Ooff --opt_for_speed=0 --use_hw_mpy=none"
		cmd += " --advice:power=all --define=__MSP430G2553__ -g --printf_support=minimal --diag_warning=225" 
		cmd += " --diag_wrap=off --display_error_number -z -m " + _name + ".map --heap_size=320 --stack_size=160"
		cmd += " -i " + Settings().mspPath + "/ccs_base/msp430/include"
		cmd += " -i " + Settings().mspPath + "/tools/compiler/ti-cgt-msp430_18.12.2.LTS/lib"
		cmd += " -i " + Settings().mspPath + "/tools/compiler/ti-cgt-msp430_18.12.2.LTS/include"
		cmd += " --reread_libs --diag_wrap=off --display_error_number --warn_sections --xml_link_info=" + _name + "_linkInfo.xml"
		cmd += " --use_hw_mpy=16 --rom_model -o " + _name + ".out ./" + file 
		cmd += " printf.obj"
		cmd +=" lnk_msp430g2553.cmd  -llibc.a" 

		result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		err = result.stderr.decode()
		out = result.stdout.decode()

		#print([out, err])

		return _name + ".out"


	def generateDummyMain(self, _callType, _numAttributes):
		code = "\nvoid main(void)\n{\n"
		code += "\tWDTCTL = WDTPW + WDTHOLD;\n"
		code += "\tDCOCTL = 0;\n"
		code += "\tBCSCTL1 = CALBC1_16MHZ;\n"
		code += "\tDCOCTL = CALDCO_16MHZ;\n\n"

		code += "\t" + _callType + " r = " + CodeGenerator().generateFunctionCall("predict", ["1.2"]*_numAttributes) + ";\n"
		code += "\tprintf(\"%i\\n\", 123);\n"
		code += "\n}\n"

		return code


	def toString(self):
		return "MSP430"