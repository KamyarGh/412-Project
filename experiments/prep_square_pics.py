import numpy as np
from scipy,misc import imread, imsave, imresize

mean = np.load('./TFD/mean.npy')
std = np.load('./TFD/std.npy')

for i in range(4):
	img = imread('~/report_visu/%d.png' % i, True)
	img = imresize(img, (48,48), 'bilinear')
	imsave('~/report_visu/%d_good.png'%i, img)