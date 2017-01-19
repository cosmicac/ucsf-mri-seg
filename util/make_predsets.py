import numpy as np
import make_datasets
import npy_to_bin 

def imgslice_to_patch_arr(imgc1, imgc2, depth):
	
	h, w, d = imgc1.shape
	patch_arr = []
	count = 0

	for i in range(h):
		for j in range(w):

			if count % 25000 == 0:
				print("Extracting patch {0}.".format(count)) 

			patch_ijc1 = make_datasets.extract_patch(imgc1, center=(i,j,depth))
			patch_ijc2 = make_datasets.extract_patch(imgc2, center=(i,j,depth))

			# concatenate the patches
			patch_ij = np.concatenate((patch_ijc1[...,np.newaxis], patch_ijc2[...,np.newaxis]), axis=3)
			patch_arr.append(patch_ij)
			count += 1

	return np.array(patch_arr) 

def predict_slice(images_and_labels, pre_images, img_number, depth, save_path):
	c1 = images_and_labels[img_number, 0, :, :, :]
	c2 = pre_images[img_number]
	img_labels = images_and_labels[img_number, 1, :, :, depth].astype('uint16').flatten()
	img = imgslice_to_patch_arr(c1, c2, depth).astype('uint16')
	npy_to_bin.flatten_and_bin(img, img_labels, save_path)


if __name__ == '__main__':

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels.npy')
	pre_images = np.load('../../data/datasets/pre_images.npy')
	n = len(images_and_labels)
	print(n)

	"""
	img8c1 = images_and_labels[8,0,:,:,:]
	img8c2 = pre_images[8]
	img8d10_labels = images_and_labels[8,1,:,:,10].astype('uint16').flatten()
	img8d10 = imgslice_to_patch_arr(img8c1, img8c2, 10).astype('uint16')

	npy_to_bin.flatten_and_bin(img8d11, img8d11_labels, '../../data/datasets/bins/img8d10_and_label_2ch_batch_1.bin')
	"""

	predict_slice(images_and_labels, pre_images, 8, 11, '../../data/datasets/bins/img8d11_and_label_2ch_batch_1.bin')

