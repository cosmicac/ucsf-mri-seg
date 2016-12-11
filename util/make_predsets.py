import numpy as np
import make_datasets

def imgslice_to_patch_arr(img, depth):
	
	h, w, d = img.shape
	patch_arr = []
	count = 0

	for i in range(h):
		for j in range(w):
			if count % 500 == 0:
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

	#img = images_and_labels[8,0,:,:,:]
	labs = images_and_labels[8,1,:,:,10]
	#patch_arr = imgslice_to_patch_arr(img, 10)
	#np.save('../../data/datasets/pred_arrs/img8d10', patch_arr
	np.save('../../data/datasets/pred_arrs/img8d10_labels', labs.flatten())