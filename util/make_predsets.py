import numpy as np
import make_datasets
import npy_to_bin 

def imgslice_to_patch_arr(img, depth):
	
	h, w, d = img.shape
	patch_arr = []
	count = 0

	for i in range(h):
		for j in range(w):
			if count % 25000 == 0:
				print("Extracting patch {0}.".format(count)) 
			patch_ij = make_datasets.extract_patch(img, center=(i,j,depth))
			patch_arr.append(patch_ij)
			count += 1

	return np.array(patch_arr) 

if __name__ == '__main__':

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels.npy')
	n = len(images_and_labels)
	print(n)

	img8 = images_and_labels[8,0,:,:,:]
	img8d11_labels = images_and_labels[8,1,:,:,11].astype('uint16').flatten()
	img8d11 = imgslice_to_patch_arr(img8, 11).astype('uint16')

	npy_to_bin.flatten_and_bin(img8d11, img8d11_labels, '../../data/datasets/bins/img8d11_and_label_batch_1.bin')

