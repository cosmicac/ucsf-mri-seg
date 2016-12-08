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

	# load images and cast as uint16
	imgs = np.load('../data/datasets/validation.npy').astype('uint16')
	labels = np.load('../data/datasets/validation_labels.npy').astype('uint16')

	# todo don't hardcode this
	imgs1 = imgs[:20000,:,:,:]
	imgs2 = imgs[20000:40000,:,:,:]
	labels1 = labels[:20000]
	labels2 = labels[20000:40000]

	flatten_and_bin(imgs1, labels1, '../data/datasets/bins/val_and_label_batch_1')
	flatten_and_bin(imgs2, labels2, '../data/datasets/bins/val_and_label_batch_2')




