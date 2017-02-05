import numpy as np 
import scipy.io as sio
import os.path
import time
import npy_to_bin
from sklearn.cluster import KMeans

NUM_SAMPLE_TRAIN = 500000
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

def sample_center_within_cluster(labels, expected_label, clusters, expected_cluster):

	# get indexes where the label matches our expected label 
	# and the cluster matches our expected cluster
	i, j, k = np.where((labels == expected_label) & (clusters == expected_cluster))

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

	# make into np arrays
	train , train_labels = np.array(train), np.array(train_labels)

	# shuffle 
	p = np.random.permutation(NUM_SAMPLE_TRAIN)
	train = train[p]
	train_labels = train_labels[p]
	
	return train, train_labels

def make_dataset_with_two_channels_kmeans(c1, c2, labs):

	n = c1.shape[0]

	train = []
	train_labels = []
	count = 1
	for i in range(n):

		# Do kmeans and find class with higher intensity
		kmeans_i = KMeans(n_clusters=2).fit(c1[i].reshape((512*512*20,1)))
		cluster_labels = kmeans_i.labels_.reshape((512,512,20))
		hi_cluster = np.argmax(kmeans_i.cluster_centers_)

		for j in range(int(NUM_SAMPLE_TRAIN/n)):

			if count % 200 == 0:
				print('extracting example {0}'.format(count))

			# generate random label and center based on radom label
			label = np.random.randint(2)
			center = sample_center_within_cluster(labs[i], label, cluster_labels, hi_cluster)

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

	# make into np arrays
	train , train_labels = np.array(train), np.array(train_labels)

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
	imgs, labels = make_dataset_with_two_channels_kmeans(c1, c2, labs)

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

	flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_kmeans_batch_1.bin')
	flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_kmeans_batch_2.bin')
	flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_kmeans_batch_3.bin')
	flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_kmeans_batch_4.bin')
	flatten_and_bin(imgs5, labels5, '../../data/datasets/bins/train_and_label_kmeans_batch_5.bin')
	
	#np.save('../../data/datasets/train_2ch_big', np.array(train))
	#np.save('../../data/datasets/train_labels_2ch_big', np.array(train_labels))

	
