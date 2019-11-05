class EpsDocument:
	def __init__(self, _width, _height):
		self.width = _width;
		self.height = _height;

		self.data = "%!PS-Adobe-3.0 EPSF-3.0\n"
		self.data += "%%CreationDate:" + "2019-07-12" + "\n"
		self.data += "%%DocumentData: Clean7Bit\n"
		self.data += "%%BoundingBox: 0 0 " + str(self.width) + " " + str(self.height) + "\n"
		self.data += "%%EndComments\n"
		self.data += "%%BeginProlog\n"
		self.data += "save 50 dict begin /q { gsave } bind def /Q { grestore } bind def /cm { 6 array astore concat } bind def /w { setlinewidth } bind def /J { setlinecap } bind def /j { setlinejoin } bind def /M { setmiterlimit } bind def /d { setdash } bind def /m { moveto } bind def /l { lineto } bind def /c { curveto } bind def /h { closepath } bind def /re { exch dup neg 3 1 roll 5 3 roll moveto 0 rlineto 0 exch rlineto 0 rlineto closepath } bind def /S { stroke } bind def /f { fill } bind def /f* { eofill } bind def /n { newpath } bind def /W { clip } bind def /W* { eoclip } bind def /BT { } bind def /ET { } bind def /EMC { mark /EMC pdfmark } bind def /rg {setrgbcolor} bind def\n";
		self.data += "%%EndProlog\n"
		self.data += "%%BeginSetup\n"
		self.data += "%%EndSetup\n"
		self.data += "%%BeginPageSetup\n"
		self.data += "%%PageBoundingBox: 0 0 " + str(self.width) + " " + str(self.height) + "\n"


	def save(self, _file):
	    self.data += "showpage\n"
	    self.data += "%%Trailer\n"
	    self.data += "end restore\n"
	    self.data += "%%EOF"

	    f = open(_file, "w")
	    f.write(self.data)
	    f.close()

	def drawCircle(self, _x, _y, _r, _col):
		self.startPath()
		self.data += str(_x) + " " + str(_y) + " " + str(_r) + " 0 360 arc "
		self.closePath();
		self.setColor(0, 0, 255)
		self.fill()

	def drawLine(self, _x0, _y0, _x1, _y1):
		self.moveTo(_x0, _y0)
		self.lineTo(_x1, _y1)
		self.stroke()

	def startPath(self):
		self.data += "n\n"

	def closePath(self):
		self.data += "h\n"

	def moveTo(self, _x, _y):
		self.data += str(_x) + " " + str(_y) + " m\n"

	def lineTo(self, _x, _y):
		self.data += str(_x) + " " + str(_y) + " l\n"

	def setColor(self, _r, _g, _b):
		self.data += str(_r/255) + " " + str(_g/255) + " " + str(_b/255) + " rg\n"

	def fill(self):
		self.data += "f\n"

	def stroke(self):
		self.data += "S\n"

	def text(self, _text, _size, _x, _y, _rot, _hAlign, _vAlign):
		self.data += "q\n/Arial findfont\n" + str(_size) + " scalefont\nsetfont\n"
		self.data += str(_x) + " " + str(_y) + " translate\n0 0 m\n(" + _text + ") false charpath flattenpath pathbbox\n4 2 roll pop pop\n"
		self.data += "0 0 m\n" + str(_rot) + " rotate\n"
		self.data += _vAlign + " mul exch " + _hAlign + " mul exch\n" # alignment
		self.data += "rmoveto\n(" + _text + ") show\nQ\n"