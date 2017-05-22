import numpy as np 
import scipy.io as sio
import os.path
import time
#import util.npy_to_bin as npy_to_bin
import npy_to_bin_fullconv
from sklearn.cluster import KMeans

NUM_SAMPLE_TRAIN = 1200
NUM_SAMPLE_TEST = 40000
PATCH_SIZE = (256, 256, 20)
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

def make_dataset_with_one_channel(c1, labs):

    n = c1.shape[0]
    
    train = np.empty((NUM_SAMPLE_TRAIN, PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2], 1), dtype='uint16') 
    train_labels = np.empty((NUM_SAMPLE_TRAIN,PATCH_SIZE[0],PATCH_SIZE[1],PATCH_SIZE[2]), dtype='uint16')
    
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
            c1patch = extract_patch(c1[i], c).reshape((PATCH_SIZE[0],PATCH_SIZE[1],PATCH_SIZE[2],1))
            lab_patch = extract_patch(labs[i], c)

            # append to datasets
            train[count,:,:,:,:] = c1patch
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

def make_fc_fullimg_valset():

    # load images and labels
    images_and_labels = np.load('../../../../data/datasets/images_and_labels.npy')
    pre_images = np.load('../../../../data/datasets/pre_images.npy')

    # validation set
    ids = np.arange(0,8)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    pre_images = pre_images[ids,:,:,:]
 
    # assert that we made a validation set
    assert images_and_labels.shape[0] == 8
    assert pre_images.shape[0] == 8
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

    imgs1 = imgs[:8,:,:,:,:]
    #imgs2 = imgs[4:,:,:,:,:]
    
    labels1 = labels[:8,:,:,:]
    #labels2 = labels[4:,:,:,:]
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../../../data/datasets/bins/val_and_label_fullimg_batch_1.bin')
    #npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../../../data/datasets/bins/val_and_label_fullimg_batch_2.bin')

def make_fc_fullimg_bme_dataset():

    # load images and labels
    images_and_labels = np.load('../../../../data/datasets/t2imgs_and_prereg_labels.npy')

    # validation set
    ids = np.arange(7,31)
    images_and_labels = images_and_labels[ids,:,:,:,:]
 
    # assert that we left out a validation set
    assert images_and_labels.shape[0] == 24

    imgs = images_and_labels[:,0,:,:,:].astype('uint16').reshape((24, 512, 512, 20, 1))
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
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../../../data/datasets/bins/train_and_label_fullimg_bme_batch_1.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../../../data/datasets/bins/train_and_label_fullimg_bme_batch_2.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs3, labels3, '../../../../data/datasets/bins/train_and_label_fullimg_bme_batch_3.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs4, labels4, '../../../../data/datasets/bins/train_and_label_fullimg_bme_batch_4.bin')

def make_fc_bme256_dataset():

    # load images and labels
    images_and_labels = np.load('../../../../data/datasets/t2imgs_and_prereg_labels.npy')
    
    # validation set
    ids = np.arange(7,31)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    
    # assert that we left out a validation set
    assert images_and_labels.shape[0] == 24
    
    # make synovitis patches and bme patches
    c1, labels = images_and_labels[:,0,:,:,:], images_and_labels[:,1,:,:,:]
    imgs, labels = make_dataset_with_one_channel(c1, labels)
    
    print(imgs.dtype)
    print(labels.dtype)
    print(imgs.shape)
    print(labels.shape)
    
    imgs1 = imgs[:300,:,:,:,:]
    imgs2 = imgs[300:600,:,:,:,:]
    imgs3 = imgs[600:900,:,:,:,:]
    imgs4 = imgs[900:,:,:,:,:]
    
    labels1 = labels[:300,:,:,:]
    labels2 = labels[300:600,:,:,:]
    labels3 = labels[600:900,:,:,:]
    labels4 = labels[900:,:,:,:]
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../../../data/datasets/bins/train_and_label_fullconv_bme256_batch_1.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../../../data/datasets/bins/train_and_label_fullconv_bme256_batch_2.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs3, labels3, '../../../../data/datasets/bins/train_and_label_fullconv_bme256_batch_3.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs4, labels4, '../../../../data/datasets/bins/train_and_label_fullconv_bme256_batch_4.bin')

def make_fc_bme256_valset():

    # load images and labels
    images_and_labels = np.load('../../../../data/datasets/t2imgs_and_prereg_labels.npy')
    
    # validation set
    ids = np.arange(0,7)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    
    # assert that we left out a validation set
    assert images_and_labels.shape[0] == 7
    
    # make synovitis patches and bme patches
    c1, labels = images_and_labels[:,0,:,:,:], images_and_labels[:,1,:,:,:]
    imgs, labels = make_dataset_with_one_channel(c1, labels)
    
    print(imgs.dtype)
    print(labels.dtype)
    print(imgs.shape)
    print(labels.shape)
    
    imgs1 = imgs[:175,:,:,:,:]
    imgs2 = imgs[175:,:,:,:,:]
    
    labels1 = labels[:175,:,:,:]
    labels2 = labels[175:,:,:,:]
    
    npy_to_bin_fullconv.flatten_and_bin(imgs1, labels1, '../../../../data/datasets/bins/val_and_label_fullconv_bme256_batch_1.bin')
    npy_to_bin_fullconv.flatten_and_bin(imgs2, labels2, '../../../../data/datasets/bins/val_and_label_fullconv_bme256_batch_2.bin')

if __name__ == '__main__':

    make_fc_bme256_dataset()
	
