import numpy as np

def flatten_and_bin(imgs, labels, save_path):
	n, h, w, d, nc = imgs.shape 

	# flatten images array
	imgs = imgs.reshape((n, h*w*d*nc))
	labels = labels.reshape((labels.shape[0], 1))

	# combine stack them horizontally
	# every row is an image
	# first element of every row is the label
	imgs_and_labels_flat = np.hstack((labels, imgs))

	print("saving to {0}".format(save_path))
	imgs_and_labels_flat.tofile(save_path)

if __name__ == '__main__':

	# load images and cast as uint16
	imgs = np.load('../../data/datasets/train_2ch_big.npy').astype('uint16')
	labels = np.load('../../data/datasets/train_labels_2ch_big.npy').astype('uint16')

	
	# todo don't hardcode this
	imgs1 = imgs[:100000,:,:,:,:]
	imgs2 = imgs[100000:200000,:,:,:,:]
	imgs3 = imgs[200000:300000,:,:,:,:]
	imgs4 = imgs[300000:400000,:,:,:,:]
	imgs5 = imgs[400000:500000,:,:,:,:]
	
	labels1 = labels[:100000]
	labels2 = labels[100000:200000]
	labels3 = labels[200000:300000]
	labels4 = labels[300000:400000]
	labels5 = labels[400000:500000]	

	flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_2ch_big_batch_1.bin')
	flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_2ch_big_batch_2.bin')
	flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_2ch_big_batch_3.bin')
	flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_2ch_big_batch_4.bin')
	flatten_and_bin(imgs5, labels5, '../../data/datasets/bins/train_and_label_2ch_big_batch_5.bin')

	#img8d9 = np.load('../../data/datasets/pred_arrs/img8d9.npy').astype('uint16')
	#img8d9_labels = np.load('../../data/datasets/pred_arrs/img8d9_labels.npy').astype('uint16')
	#flatten_and_bin(img8d9, img8d9_labels, '../../data/datasets/bins/img8d9_and_label_batch_1.bin')
	

