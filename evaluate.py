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
			samples[i,...].squeeze() if not flat_samples else np.reshape(samples[i,...].squeeze(), img_shape)
		)


def save_dash_samples(catalog, samples, index, save_dir, flat_samples=True, img_shape=None, num_to_save=10):
	fig = plt.figure()
	fname = '%d.png' % index
	for i in range(min(num_to_save, samples.shape[0])):
		ax = fig.add_subplot(1, num_to_save, i+1)
		ax.imshow(samples[i,...].squeeze() if not flat_samples else np.reshape(samples[i,...].squeeze(), img_shape), cmap='Greys_r')

	plt.savefig(
		os.path.join(
			save_dir,
			fname
		)
	)
	plt.close()

	catalog.write('{},image,{}\n'.format(fname, 'Val @ %d:' % index))
	catalog.flush()


def save_ae_samples(catalog, val_batch, noisy_val_batch, val_recon, index,
					save_dir, num_to_save=5, save_gray=True):
	fig = plt.figure()
	fname = '%d.png' % index
	for i in range(min(num_to_save, val_batch.shape[0])):
		ax = fig.add_subplot(3, num_to_save, i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		if save_gray:
			ax.imshow(noisy_val_batch[i,...].squeeze(), cmap='Greys_r')
		else:
			ax.imshow(noisy_val_batch[i,...].squeeze())

		ax = fig.add_subplot(3, num_to_save, i+1+num_to_save)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		if save_gray:
			ax.imshow(val_batch[i,...].squeeze(), cmap='Greys_r')
		else:
			ax.imshow(val_batch[i,...].squeeze())

		ax = fig.add_subplot(3, num_to_save, i+1+2*num_to_save)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		if save_gray:
			ax.imshow(val_recon[i,...].squeeze(), cmap='Greys_r')
		else:
			ax.imshow(val_recon[i,...].squeeze())

	fig.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(
		os.path.join(
			save_dir,
			fname
		)
	)
	plt.close()

	catalog.write('{},image,{}\n'.format(fname, 'Val @ %d:' % index))
	catalog.flush()