import numpy as np
from scipy.misc import imread, imsave, imresize

mean = np.load('./TFD/mean.npy')
std = np.load('./TFD/std.npy')

for i in range(5):
	img = imread('/u/kamyar/report_visu/%d.png' % i, True)
	img = imresize(img, (48,48), 'bilinear')
	img = (img - mean) / std
	imsave('/u/kamyar/report_visu/%d_norm.png'%i, img)