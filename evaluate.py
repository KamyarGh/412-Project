import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process
from scipy.misc import imsave
import os

def new_process(func):
	def wrapper(*args):
		Process(target=func, args=(args)).start()
	return wrapper

# @new_process
def save_samples(samples, index, save_dir, flat_samples=True, img_shape=None, num_to_save=10):
	for i in range(min(num_to_save, samples.shape[0])):
		imsave(
			os.path.join(
				save_dir,
				'%d_%d.png' % (index, i)
			),
			samples[i,:,:].squeeze() if not flat_samples else np.reshape(samples[i,:].squeeze(), img_shape)
		)

def save_dash_samples(catalog, samples, index, save_dir, flat_samples=True, img_shape=None, num_to_save=10):
	fig = plt.figure()
	fname = '%d.png' % index
	for i in range(min(num_to_save, samples.shape[0])):
		ax = fig.add_subplot(1, num_to_save, i+1)
		ax.imshow(samples[i,:,:].squeeze() if not flat_samples else np.reshape(samples[i,:].squeeze(), img_shape), cmap='Greys_r')

	plt.savefig(
		os.path.join(
			save_dir,
			fname
		)
	)
	plt.close()

	catalog.write('{},image,{}\n'.format(fname, 'Val @ %d:' % index))
	catalog.flush()