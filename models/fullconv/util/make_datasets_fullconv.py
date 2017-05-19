import numpy as np 
import scipy.io as sio
import os.path
import time
#import util.npy_to_bin as npy_to_bin
import npy_to_bin_fullconv
from sklearn.cluster import KMeans

NUM_SAMPLE_TRAIN = 2400
NUM_SAMPLE_TEST = 40000
PATCH_SIZE = (128, 128, 16)
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

def sample_centers(labels, expected_labels):

    # map the possible labels to the possible center indexes
    labs_to_centers = {}
    for l in np.unique(expected_labels):
        labs_to_centers[l] = np.where((labels == l))
    
    # randomly select patch centers
    centers = []
    for el in expected_labels:
        i, j, k = labs_to_centers[el] 
        ri = np.random.randint(0, len(i))
        centers.append((i[ri], j[ri], k[ri]))
    
    return centers

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
    
    train = np.empty((NUM_SAMPLE_TRAIN,128,128,16,2), dtype='uint16') 
    train_labels = np.empty((NUM_SAMPLE_TRAIN,128,128,16), dtype='uint16')
    
    count = 0
    for i in range(n):

        # generate random label and center based on radom label
        center_labs = np.random.randint(2, size=int(NUM_SAMPLE_TRAIN/n))
        centers = sample_centers(labs[i], center_labs)
        
        # extract patches for each center
        for c in centers:

            if count % 200 == 0:
                print('extracting example {0}'.format(count))

            # extract patches from channels
            c1patch = extract_patch(c1[i], c)
            c2patch = extract_patch(c2[i], c)
            lab_patch = extract_patch(labs[i], c)
            
            # concatenate the patches
            patch = np.concatenate((c1patch[...,np.newaxis], c2patch[...,np.newaxis]), axis=3)

            # append to datasets
            train[count,:,:,:,:] = patch
            train_labels[count,:,:,:] = lab_patch

	    # increment counter
            count += 1

    # make into np arrays
    train , train_labels = np.array(train).astype('uint16'), np.array(train_labels).astype('uint16')

    # shuffle 
    p = np.random.permutation(NUM_SAMPLE_TRAIN)
    train = train[p]
    train_labels = train_labels[p]
	
    return train, train_labels

"""
Input: 	full 512x512x20 image labels, 
		sequence of expected labels (sequence of 0's and 1's)
		full 512x512x20 cluster labels
		the expected cluster to take
output: sequence of patch centers that match the input sequence of expected_labels
"""
def sample_centers_within_cluster(labels, expected_labels, clusters, expected_cluster):

	# map the possible labels to the possible center indexes 
	labs_to_centers = {}
	for l in np.unique(expected_labels):
		labs_to_centers[l] = np.where((labels == l) & (clusters == expected_cluster))

	# randomly select patch centers
	centers = []
	for el in expected_labels:
		i, j, k = labs_to_centers[el]
		ri = np.random.randint(0, len(i))
		centers.append((i[ri], j[ri], k[ri]))

	return centers

def make_dataset_with_one_channels_kmeans_t2(c1, labs):

	n = c1.shape[0]

	train = np.empty((NUM_SAMPLE_TRAIN,32,32,8,1), dtype='uint16')
	train_labels = []
	count = 0

	print(labs.shape)
	for i in range(n):

		# Do kmeans and find class with higher intensity
		kmeans_i = KMeans(n_clusters=2).fit(c1[i].reshape((512*512*20,1)))
		cluster_labels = kmeans_i.labels_.reshape((512,512,20))
		hi_cluster = np.argmax(kmeans_i.cluster_centers_)

		# generate labels 50/50 positive negative
		train_labels_i = np.random.randint(1, size=int(NUM_SAMPLE_TRAIN/n))
		train_labels.extend(train_labels_i)

		# get centers that correspond to generated labels
		centers = sample_centers_within_cluster(labs[i], train_labels_i, cluster_labels, hi_cluster)

		# go and extract the patches for each center and add to our training array
		for c in centers:

			if count % 200 == 0:
				print('extracting healthy example {0}'.format(count))

			# extract patches from channels
			patch = extract_patch(c1[i], c).reshape((32,32,8,1))

			# append to datasets
			train[count,:,:,:,:] = patch

			# increment counter
			count += 1

	# make into np arrays
	train_labels = np.array(train_labels, dtype='uint16')
	print(train.dtype)
	print(train_labels.dtype)
	
	return train, train_labels



def make_dataset_with_two_channels_kmeans(c1, c2, labs):

	n = c1.shape[0]

	train = np.empty((NUM_SAMPLE_TRAIN,32,32,8,2), dtype='uint16')
	train_labels = []
	count = 0

	print(labs.shape)
	for i in range(n):

		# Do kmeans and find class with higher intensity
		kmeans_i = KMeans(n_clusters=2).fit(c1[i].reshape((512*512*20,1)))
		cluster_labels = kmeans_i.labels_.reshape((512,512,20))
		hi_cluster = np.argmax(kmeans_i.cluster_centers_)

		# generate labels 50/50 positive negative
		train_labels_i = np.random.randint(2, size=int(NUM_SAMPLE_TRAIN/n))
		train_labels.extend(train_labels_i)

		# get centers that correspond to generated labels
		centers = sample_centers_within_cluster(labs[i], train_labels_i, cluster_labels, hi_cluster)

		# go and extract the patches for each center and add to our training array
		for c in centers:

			if count % 200 == 0:
				print('extracting syn example {0}'.format(count))

			# extract patches from channels
			c1patch = extract_patch(c1[i], c)
			c2patch = extract_patch(c2[i], c)

			# concatenate the patches
			patch = np.concatenate((c1patch[...,np.newaxis], c2patch[...,np.newaxis]), axis=3)
			# append to datasets
			train[count,:,:,:,:] = patch

			# increment counter
			count += 1

	# make into np arrays
	train_labels = np.array(train_labels, dtype='uint16')
	print(train.dtype)
	print(train_labels.dtype)
	
	return train, train_labels

def make_bme_dataset_1c(c1, merged_labels):

	n = c1.shape[0]

	train = []
	train_labels = []
	count = 1

	for i in range(n):

		labels = merged_labels[i]

		# find centers that are bme
		centers =  zip(*np.where(labels == 1))

		# loop through and make extract the patches
		for c in centers:

			if count % 200 == 0: 
				print('extracting bme example {0}'.format(count))
	
			# extract patch
			patch = extract_patch(c1[i], c).reshape((32,32,8,1))

			# append patches
			train.append(patch)
			train_labels.append(1)

			count += 1

	# make into np arrays
	train, train_labels = np.array(train, dtype='uint16'), np.array(train_labels, dtype='uint16')
	print(train.dtype)
	print(train_labels.dtype)

	return train, train_labels

def make_bme_dataset(c1, c2, merged_labels):
	
	n = c1.shape[0]

	train = []
	train_labels = []
	count = 1

	for i in range(n):

		labels = merged_labels[i]

		# find centers that are bme
		centers =  zip(*np.where(labels == 2))

		# loop through and make extract the patches
		for c in centers:

			if count % 200 == 0: 
				print('extracting bme example {0}'.format(count))
	
			# extract patch
			c1patch = extract_patch(c1[i], c)
			c2patch = extract_patch(c2[i], c)

			# concatenate patches
			patch = np.concatenate((c1patch[...,np.newaxis], c2patch[...,np.newaxis]), axis=3)

			# append patches
			train.append(patch)
			train_labels.append(2)

			count += 1

	# make into np arrays
	train, train_labels = np.array(train, dtype='uint16'), np.array(train_labels, dtype='uint16')
	print(train.dtype)
	print(train_labels.dtype)

	return train, train_labels

def maket2dataset():

	# load images and labels
	images_and_labels = np.load('../../data/datasets/t2imgs_and_prereg_labels.npy')

	# validation set 
	ids = (np.arange(6, 29))
	images_and_labels = images_and_labels[ids,:,:,:,:]

	# assert validation set
	assert images_and_labels.shape[0] == 23

	# make bme patches and healthy patches
	c1, labels = images_and_labels[:,0,:,:,:], images_and_labels[:,1,:,:,:]
	bme_imgs, bme_labels = make_bme_dataset_1c(c1, labels)
	h_imgs, h_labels = make_dataset_with_one_channels_kmeans_t2(c1, labels)

	# combine the two
	imgs = np.concatenate((h_imgs, bme_imgs), axis=0)
	labels = np.concatenate((h_labels, bme_labels), axis=0)

	# shuffle the dataset
	print("shuffling")
	print(imgs.shape)
	print(labels.shape)
	p = np.random.permutation(imgs.shape[0])
	imgs = imgs[p]
	labels = labels[p]
	print("done shuffling")

	print(imgs.dtype)
	print(labels.dtype)
	print(imgs.shape)
	print(labels.shape)

	imgs1 = imgs[:100000,:,:,:,:]
	imgs2 = imgs[100000:200000,:,:,:,:]
	imgs3 = imgs[200000:300000,:,:,:,:]
	imgs4 = imgs[300000:,:,:,:,:]
	#imgs5 = imgs[400000:,:,:,:,:]
	#imgs6 = imgs[500000:,:,:,:,:]
	
	labels1 = labels[:100000]
	labels2 = labels[100000:200000]
	labels3 = labels[200000:300000]
	labels4 = labels[300000:]
	#labels5 = labels[400000:]
	#labels6 = labels[500000:]	

	npy_to_bin.flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_t2bmeonly_batch_1.bin')
	npy_to_bin.flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_t2bmeonly_batch_2.bin')
	npy_to_bin.flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_t2bmeonly_batch_3.bin')
	npy_to_bin.flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_t2bmeonly_batch_4.bin')
	#npy_to_bin.flatten_and_bin(imgs5, labels5, '../../data/datasets/bins/train_and_label_t2bmeonly_batch_5.bin')

def maket1dataset():

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels_bmesyn_regfix.npy')
	pre_images = np.load('../../data/datasets/pre_images_aligned_regfix.npy')

	# valdiation set
	ids = np.concatenate((np.arange(5), np.arange(11,29)))
	images_and_labels = images_and_labels[ids,:,:,:,:]
	pre_images = pre_images[ids,:,:,:]

	# assert that we left out a validation set
	assert images_and_labels.shape[0] == 23
	assert pre_images.shape[0] == 23
	assert pre_images.shape[0] == images_and_labels.shape[0]

	# make synivotis patches and bme patches
	c1, c2, labels = images_and_labels[:,0,:,:,:], pre_images, images_and_labels[:,1,:,:,:]
	bme_imgs, bme_labels = make_bme_dataset(c1, c2, labels)
	syn_imgs, syn_labels = make_dataset_with_two_channels_kmeans(c1, c2, labels)
	
	# combine the two
	imgs = np.concatenate((syn_imgs, bme_imgs), axis=0)
	labels = np.concatenate((syn_labels, bme_labels), axis=0)

	# shuffle the dataset
	print("shuffling")
	print(imgs.shape)
	print(labels.shape)
	p = np.random.permutation(imgs.shape[0])
	imgs = imgs[p]
	labels = labels[p]
	print("done shuffling")


	print(imgs.dtype)
	print(labels.dtype)
	print(imgs.shape)
	print(labels.shape)

	imgs1 = imgs[:100000,:,:,:,:]
	imgs2 = imgs[100000:200000,:,:,:,:]
	imgs3 = imgs[200000:300000,:,:,:,:]
	imgs4 = imgs[300000:400000,:,:,:,:]
	imgs5 = imgs[400000:,:,:,:,:]
	#imgs6 = imgs[500000:,:,:,:,:]
	
	labels1 = labels[:100000]
	labels2 = labels[100000:200000]
	labels3 = labels[200000:300000]
	labels4 = labels[300000:400000]
	labels5 = labels[400000:]
	#labels6 = labels[500000:]	

	npy_to_bin.flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_regfix_batch_1.bin')
	npy_to_bin.flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_regfix_batch_2.bin')
	npy_to_bin.flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_regfix_batch_3.bin')
	npy_to_bin.flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_regfix_batch_4.bin')
	npy_to_bin.flatten_and_bin(imgs5, labels5, '../../data/datasets/bins/train_and_label_regfix_batch_5.bin')
	#npy_to_bin.flatten_and_bin(imgs6, labels6, '../../data/datasets/bins/train_and_label_bmet1postreg_batch_6.bin')
	

	#np.save('../../data/datasets/train_2ch_big', np.array(train))
	#np.save('../../data/datasets/train_labels_2ch_big', np.array(train_labels))

def make_fc_t1dataset():

    # load images and labels
    images_and_labels = np.load('../../data/datasets/images_and_labels.npy')
    pre_images = np.load('../../data/datasets/pre_images.npy')
    
    # validation set
    ids = np.arange(8,32)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    pre_images = pre_images[ids,:,:,:]
    
    # assert that we left out a validation set
    assert images_and_labels.shape[0] == 24
    assert pre_images.shape[0] == 24
    assert pre_images.shape[0] == images_and_labels.shape[0]
    
    # make synovitis patches and bme patches
    c1, c2, labels = images_and_labels[:,0,:,:,:], pre_images, images_and_labels[:,1,:,:,:]
    imgs, labels = make_dataset_with_two_channels(c1, c2, labels)
    
    print(imgs.dtype)
    print(labels.dtype)
    print(imgs.shape)
    print(labels.shape)
    
    imgs1 = imgs[:600,:,:,:,:]
    imgs2 = imgs[600:1200,:,:,:,:]
    imgs3 = imgs[1200:1800,:,:,:,:]
    imgs4 = imgs[1800:,:,:,:,:]
    
    labels1 = labels[:600,:,:,:]
    labels2 = labels[600:1200,:,:,:]
    labels3 = labels[1200:1800,:,:,:]
    labels4 = labels[1800:,:,:,:]
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../data/datasets/bins/train_and_label_fullconv_batch_1.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../data/datasets/bins/train_and_label_fullconv_batch_2.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs3, labels3, '../../data/datasets/bins/train_and_label_fullconv_batch_3.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs4, labels4, '../../data/datasets/bins/train_and_label_fullconv_batch_4.bin')

def make_fc_fullimg_dataset():

    # load images and labels
    images_and_labels = np.load('../../../../data/datasets/images_and_labels.npy')
    pre_images = np.load('../../../../data/datasets/pre_images.npy')

    # validation set
    ids = np.arange(8,32)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    pre_images = pre_images[ids,:,:,:]
 
    # assert that we left out a validation set
    assert images_and_labels.shape[0] == 24
    assert pre_images.shape[0] == 24
    assert pre_images.shape[0] == images_and_labels.shape[0]

    c1, c2 = images_and_labels[:,0,:,:,:], pre_images
    print(c1.shape)
    print(c2.shape)
    imgs = np.concatenate((c1[...,np.newaxis], c2[...,np.newaxis]), axis=4).astype('uint16')
    labels = images_and_labels[:,1,:,:,:].astype('uint16')

    print(imgs.shape)
    print(labels.shape)
    print(imgs.dtype)
    print(labels.dtype)

    imgs1 = imgs[:6,:,:,:,:]
    imgs2 = imgs[6:12,:,:,:,:]
    imgs3 = imgs[12:18,:,:,:,:]
    imgs4 = imgs[18:,:,:,:,:]
    
    labels1 = labels[:6,:,:,:]
    labels2 = labels[6:12,:,:,:]
    labels3 = labels[12:18,:,:,:]
    labels4 = labels[18:,:,:,:]
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../../../data/datasets/bins/train_and_label_fullimg_batch_1.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../../../data/datasets/bins/train_and_label_fullimg_batch_2.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs3, labels3, '../../../../data/datasets/bins/train_and_label_fullimg_batch_3.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs4, labels4, '../../../../data/datasets/bins/train_and_label_fullimg_batch_4.bin')
 
if __name__ == '__main__':

    make_fc_fullimg_dataset()
	
