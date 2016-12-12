import numpy as np

def flatten_and_bin(imgs, labels, save_path):
	n, h, w, d = imgs.shape 

	# flatten images array
	imgs = imgs.reshape((n, h*w*d))
	labels = labels.reshape((labels.shape[0], 1))

	# combine stack them horizontally
	# every row is an image
	# first element of every row is the label
	imgs_and_labels_flat = np.hstack((labels, imgs))

	print("saving to {0}".format(save_path))
	imgs_and_labels_flat.tofile(save_path)

if __name__ == '__main__':

	"""
	# load images and cast as uint16
	imgs = np.load('../../data/datasets/train_mix.npy').astype('uint16')
	labels = np.load('../../data/datasets/train_labels_mix.npy').astype('uint16')

	
	# todo don't hardcode this

	
	imgs1 = imgs[:20000,:,:,:]
	imgs2 = imgs[20000:40000,:,:,:]
	imgs3 = imgs[40000:60000,:,:,:]
	imgs4 = imgs[60000:80000,:,:,:]
	imgs5 = imgs[80000:100000,:,:,:]
	labels1 = labels[:20000]
	labels2 = labels[20000:40000]
	labels3 = labels[40000:60000]
	labels4 = labels[60000:80000]
	labels5 = labels[80000:100000]	

	flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_mix_batch_1.bin')
	flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_mix_batch_2.bin')
	flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_mix_batch_3.bin')
	flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_mix_batch_4.bin')
	flatten_and_bin(imgs5, labels5, '../../data/datasets/bins/train_and_label_mix_batch_5.bin')
	"""

	img8d9 = np.load('../../data/datasets/pred_arrs/img8d9.npy').astype('uint16')
	img8d9_labels = np.load('../../data/datasets/pred_arrs/img8d9_labels.npy').astype('uint16')
	flatten_and_bin(img8d9, img8d9_labels, '../../data/datasets/bins/img8d9_and_label_batch_1.bin')



