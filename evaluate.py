import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from scipy.misc import imsave
import os

def new_process(func):
	def wrapper(*args):
		Process(target=func, args=(args)).start()
	return wrapper

@new_process
def save_samples(samples, index, save_dir, flat_samples=True, img_shape=None, num_to_save=10):
	for i in range(min(num_to_save, samples.shape[0])):
		imsave(
			os.path.join(
				save_dir,
				'%d_%d.png' % (index, i)
			),
			samples[i,:,:].squeeze() if not flat_samples else np.reshape(samples[i,:].squeeze(), img_shape)
		)