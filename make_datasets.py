import numpy as np 
import scipy.io as sio
import os.path
import tensorflow as tf

NUM_SAMPLE_TRAIN = 100
NUM_SAMPLE_TEST = 100
NUM_SAMPLES_EPOCH_TRAIN = 10
PATCH_SIZE = (32, 32, 8)
PAD_SIZE = int(max(PATCH_SIZE)/2 + 2)

TIMES = ['Baseline', '1_Month', '3_Month', '1_Year']
IMAGE_PATH_TEMPLATE = '../data/raw_images/post_{0}_{1}.mat'
LABEL_PATH_TEMPLATE = '../data/merged_labels/{0}_{1}.npy'

def extract_patch(img, labels, expected_label):

		# height, width, and depth of original image
		h, w, d = img.shape

		# pad image according to patch size
		pad_img = np.pad(img, PAD_SIZE, zeropad)

		# sample center of patch in original image at random
		ch, cw, cd = sample_center(labels, expected_label)

		# center of our patch in the padded image
		chp, cwp, cdp = ch + PAD_SIZE, cw + PAD_SIZE, cd + PAD_SIZE

		# patch margins
		phm, pwm, pdm = PATCH_SIZE[0]/2, PATCH_SIZE[1]/2, PATCH_SIZE[2]/2

		# extract patch
		patch = pad_img[chp-phm:chp+phm, cwp-pwm:cwp+pwm, cdp-pdm:cdp+pdm]

		#extract label
		label = labels[ch, cw, cd]

		return patch

def sample_center(labels, expected_label):

	# get indexes where the label matches our expected label
	i, j, k = np.where(labels == expected_label)
	idx = list(zip(i,j,k))

	# sample an index at random and return it as a tuple
	ridx = np.random.randint(0, len(idx))

	return idx[ridx]


def zeropad(vector, pad_width, iaxis, kwargs):
	vector[:pad_width[0]] = 0
	vector[-pad_width[1]:] = 0
	return vector

if __name__ == '__main__':

	"""
	# load images into memory
	images_and_labels = []
	for i in range(1, 18):
		for time in TIMES:

			# construct image and label file paths
			image_file = IMAGE_PATH_TEMPLATE.format(i, time)
			label_file = LABEL_PATH_TEMPLATE.format(i, time)

			# only load if both the image and labels exist
			if os.path.isfile(image_file) and os.path.isfile(label_file):
				print("Loading image for patient {0} at time {1}.".format(i, time))
				image = sio.loadmat(image_file)['ImageWTR']
				labels = np.load(label_file)

				# don't sample from the earlier 256x256x20 images
				if image.shape[0] == 512:
					images_and_labels.append((image, labels))

	print(len(images_and_labels))
	np.save('../data/datasets/images_and_labels', images_and_labels)
	"""

	
	images_and_labels = np.load('../data/datasets/images_and_labels.npy')
	n = len(images_and_labels)
	
	train = []
	train_labels = []
	
	# extract positive examples
	for i in range(int(NUM_SAMPLE_TRAIN/2)):
		print("extracting positive example {0}".format(i))
		# pick image at random
		img_and_label = images_and_labels[np.random.randint(0, n)]
		patch = extract_patch(img_and_label[0], img_and_label[1], 1)
		train.append(patch)
		train_labels.append(1)

	# extract negative examples
	for i in range(int(NUM_SAMPLE_TRAIN/2)):
		print('extracting negative example {0}'.format(i))
		# pick image at random
		img_and_label = images_and_labels[np.random.randint(0, n)]
		patch = extract_patch(img_and_label[0], img_and_label[1], 0)
		train.append(patch)
		train_labels.append(0)

	print("saving training set")
	np.save('../data/datasets/train_test', np.array(train))
	np.save('../data/datasets/train_test_labels', np.array(train_labels))

	