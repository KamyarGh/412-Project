import numpy as np
import os
from json import dump
from scipy.misc import imsave

IPC = 10000

cifar_dir = '/ais/gobi3/u/yujiali/temp/kamyar/cifar-10-batches-py'
save_dir = '/ais/gobi3/u/yujiali/temp/kamyar/CIFAR_10'

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


for i in range(4):
	print(i+1)

	dict = unpickle(os.path.join(cifar_dir, 'data_batch_%d'%(i+1)))
	data = dict['data']
	labels = dict['labels']

	for j, img in enumerate(data):
		if j % 1000 == 0: print('\t%d' % j)

		img = np.reshape(img, [32, 32, 3], order='F')
		img = np.transpose(img, (1,0,2))
		imsave(os.path.join(save_dir, 'train', 'patches', '%d.png' % (i*IPC + j)), img)

		j_dict = {
			'patch': os.path.join(save_dir, 'train', 'patches', '%d.png' % (i*IPC + j)),
			'label': labels[j]
		}
		dump(j_dict, open(os.path.join(save_dir, 'train', 'info', '%d.json' % (i*IPC + j)), 'w'))

	dict = unpickle(os.path.join(cifar_dir, 'data_batch_%d'%(i+1)))
	data = dict['data']
	labels = dict['labels']


i = 4
print(i+1)

dict = unpickle(os.path.join(cifar_dir, 'data_batch_%d'%(i+1)))
data = dict['data']
labels = dict['labels']

for j, img in enumerate(data):
	if j % 1000 == 0: print('\t%d' % j)

	img = np.reshape(img, [32, 32, 3], order='F')
	img = np.transpose(img, (1,0,2))
	imsave(os.path.join(save_dir, 'valid', 'patches', '%d.png' % (i*IPC + j)), img)

	j_dict = {
		'patch': os.path.join(save_dir, 'valid', 'patches', '%d.png' % (i*IPC + j)),
		'label': labels[j]
	}
	dump(j_dict, open(os.path.join(save_dir, 'valid', 'info', '%d.json' % (i*IPC + j)), 'w'))