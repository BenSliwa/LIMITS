import numpy as np

class ConfusionMatrix:
	def __init__(self):
		self.classes = []
		self.data = []


	def init(self, _classes):
		self.classes = _classes
		s = len(_classes)
		self.data = np.zeros([s, s], dtype = int) 


	def update(self, _prediction, _label):
		p = self.classes.index(_prediction)
		l = self.classes.index(_label)

		self.data[l][p] += 1


	def merge(self, _matrix): # here it is assumed that both matrices have the same column mapping
		l = len(self.classes)
		if l==0:
			self.classes = _matrix.classes
			self.data = _matrix.data
		else:
			for y in range(l):
				for x in range(l):
					self.data[y][x] += _matrix.data[y][x]


	def calc(self):
		l = self.data.shape[0]
		totalInstances = self.data.sum()
		TP = []
		FP = []
		FN = []
		for i in range(0, l): # for all classes
			nInstances = np.sum(self.data[i, :])
			tp = self.data[i,i]
			fp = np.sum(self.data[:, i]) - tp
			fn = nInstances - tp
			
			weight = nInstances / totalInstances
			TP.append(weight * tp)
			FP.append(weight * fp)
			FN.append(weight * fn)

		TP = np.sum(TP)
		FP = np.sum(FP)
		FN = np.sum(FN)

		accuracy = self.data.trace() / totalInstances * 100
		precision =	TP/(TP+FP) * 100
		recall = TP/(TP+FN) * 100
		f_score = 2 * precision * recall / (precision + recall) 

		return accuracy, precision, recall, f_score


	def save(self, _file):
		np.savetxt(_file, np.asmatrix(self.data), delimiter=',', fmt='%i', header=",".join(self.classes), comments='')
