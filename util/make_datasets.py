import numpy as np 
import scipy.io as sio
import os.path
import time

NUM_SAMPLE_TRAIN = 100000
NUM_SAMPLE_TEST = 40000
PATCH_SIZE = (32, 32, 8)
PATCH_MARGINS = (int(PATCH_SIZE[0]/2), int(PATCH_SIZE[1]/2), int(PATCH_SIZE[2]/2))

def extract_patch(img, center):

	# height, width, and depth of original image
	h, w, d = img.shape

	# unpack center
	ch, cw, cd = center

	# calculate the border indexes of the original image and our patch
	ihl, ihu, phl, phu = calc_borders_1d(ch, h, PATCH_SIZE[0], PATCH_MARGINS[0]) 
	iwl, iwu, pwl, pwu = calc_borders_1d(cw, w, PATCH_SIZE[1], PATCH_MARGINS[1])
	idl, idu, pdl, pdu = calc_borders_1d(cd, d, PATCH_SIZE[2], PATCH_MARGINS[2])

	# initalize patch
	patch = np.zeros(PATCH_SIZE, dtype='uint16')

	# extract patch
	patch[phl:phu, pwl:pwu, pdl:pdu] = img[ihl:ihu, iwl:iwu, idl:idu]

	return patch

# takes an center index, and returns the border indexes of
# both the original image and new patch along one dimension
def calc_borders_1d(center, image_max, patch_max, patch_margin):

	# image_min is assumed to be 0
	image_lower = center - patch_margin
	patch_lower = 0

	# cannot start from a negative image index
	# so, just take what you can and adjust patch index
	if image_lower < 0:
		patch_lower = -image_lower
		image_lower = 0

	# cannot index past the image size
	# so, just take what you can and adjust patch index
	image_upper = center + patch_margin
	patch_upper = patch_max
	diff_upper = image_upper - image_max
	if diff_upper > 0:
		patch_upper = patch_max - diff_upper
		image_upper = image_max

	return image_lower, image_upper, patch_lower, patch_upper



def sample_center(labels, expected_label):

	# get indexes where the label matches our expected label
	i, j, k = np.where(labels == expected_label)

	# sample an index at random and return it as a tuple
	ri = np.random.randint(0, len(i))

	return i[ri], j[ri], k[ri]

def make_dataset_one_channel():

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels.npy')

	n = len(images_and_labels)
	print(n)
	
	train = []
	train_labels = []

	for i in range(int(NUM_SAMPLE_TRAIN)):

		if i % 200 == 0:
			print('extracting example {0}'.format(i))

		# pick image at random
		img_and_label = images_and_labels[np.random.randint(0, n)]

		# pick a label, 0 or 1, at random
		rlabel = np.random.randint(2)

		# extract patch
		patch = extract_patch(img_and_label[0], labels=img_and_label[1], expected_label=rlabel)
		train.append(patch)
		train_labels.append(rlabel)

	print("saving training set")
	np.save('../../data/datasets/train_mix', np.array(train))
	np.save('../../data/datasets/train_labels_mix', np.array(train_labels))

def make_dataset_with_two_channels(c1, c2, labs):

	n = c1.shape[0]

	train = []
	train_labels = []
	count = 1
	for i in range(n):
		for j in range(int(NUM_SAMPLE_TRAIN/n)):

			if count % 200 == 0:
				print('extracting example {0}'.format(count))

			# generate random label and center based on radom label
			label = np.random.randint(2)
			center = sample_center(labs[i], label)

			# extract patches from channels
			c1patch = extract_patch(c1[i], center)
			c2patch = extract_patch(c2[i], center)

			# concatenate the patches
			patch = np.concatenate((c1patch[...,np.newaxis], c2patch[...,np.newaxis]), axis=3)
			# append to datasets
			train.append(patch)
			train_labels.append(label)

			# increment counter
			count += 1

	# shuffle 
	p = np.random.permutation(NUM_SAMPLE_TRAIN)
	train = train[p]
	train_labels = train_labels[p]
	
	return train, train_labels

if __name__ == '__main__':

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels.npy')
	pre_images = np.load('../../data/datasets/pre_images.npy')

	c1, c2, labs = images_and_labels[:,0,:,:,:], pre_images, images_and_labels[:,1,:,:,:]
	train, train_labels = make_dataset_with_two_channels(c1, c2, labs)

	np.save('../../data/datasets/train_2ch', np.array(train))
	np.save('../../data/datasets/train_labels_2ch', np.array(train_labels))


	
