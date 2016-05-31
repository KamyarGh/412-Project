import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_values(path):
	with open(path, 'r') as f:
		f.readline()
		X, Y = [], []
		for line in f:
			line = line.split(',')
			X.append(float(line[0]))
			Y.append(float(line[1]))
	return np.array(X), np.array(Y)

