import numpy as np
import sys
import os
from scipy.io import loadmat
from scipy.misc import imsave

if __name__ == '__main__':
	data_dir = sys.argv[1]
	save_dir = sys.argv[2]
	split = {}
	split['train'] = float(sys.argv[3])
	split['val'] = float(sys.argv[4])
	split['test'] = 1 - split['train'] - split['val']

	d = loadmat(data_dir)
	imgs = d['images'].astype(float)

	num_imgs = imgs.shape[0]

	mean = np.mean(imgs, axis = 0)
	std = np.std(imgs, axis = 0)

	print('Got mean and std!')

	for i in xrange(int(np.ceil(split['train'] * num_imgs))):
		np.save(
			os.path.join(save_dir, 'train/%d' % i),
			(imgs[i,:,:] - mean) / std
		)
		if i % 1000 == 0: print('%d Saved...' % i)

	for i in xrange(int(np.ceil(split['train'] * num_imgs)), num_imgs):
		np.save(
			os.path.join(save_dir, 'val/%d' % (i - np.ceil(split['train'] * num_imgs))),
			(imgs[i,:,:] - mean) / std
		)
		if i % 1000 == 0: print('%d Saved...' % i)

	print('Saving mean and std!')
	np.save(
		os.path.join(save_dir, 'mean'),
		mean
	)
	np.save(
		os.path.join(save_dir, 'std'),
		std
	)