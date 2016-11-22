import numpy as np 
import scipy.io as sio
import os.path
import tensorflow as tf

TIMES = ['Baseline', '1_Month', '3_Month', '1_Year']
PATH_TEMPLATE = '../data/raw_images/post_{0}_{1}.mat'
NUM_SAMPLES_EPOCH_TRAIN = 10
PATCH_SIZE = (32, 32, 8)
PAD_SIZE = max(PATCH_SIZE)/2 + 2


if __name__ == '__main__':

	# load images into memory
	images = []
	for i in range(1, 18):
		for time in TIMES:
			file = PATH_TEMPLATE.format(i, time)
			if os.path.isfile(file):
				print(file)
				image = sio.loadmat(file)['ImageWTR']
				print(type(image))
				image = tf.cast(image, tf.float32)
				print(type(image))
				images.append(image)
	
	for i in range(NUM_SAMPLES_EPOCH_TRAIN)
		# pick image at random
		img = images[np.random.randint(0,40)]


def extract_patch(img, labels, expected_label):

		# height, width, and depth of original image
		h, w, d = img.shape

		# pad image according to patch size
		pad_img = np.pad(img, PAD_SIZE, zeropad)

		# center of patch in original image
		ch, cw, cd = np.random.randint(0, h),
					 np.random.randint(0, w),
					 np.random.randint(0, d)

		# center of our patch in the padded image
		chp, cwp, cdp = chp + PAD_SIZE, cw + PAD_SIZE, cd + PAD_SIZE

		# patch margins
		phm, pwm, pdm = PATCH_SIZE[0]/2, PATCH_SIZE[1]/2, PATCH_SIZE[2]/2

		# extract patch
		patch = pad_img[chp-phm:chp+phm, cwp-pwm:cwp+pwm, cdp-pdm:cdp+pdm]

		#extract label
		label = labels[ch, cw, cd]

		return patch



def zeropad(vector, pad_width, iaxis, kwargs):
	vector[:pad_width[0]] = 0
	vector[-pad_width[1]:] = 0
	return vector




	#image = sio.loadmat('../data/raw_images/post_2_3_Month.mat')
	#labels = np.asarray(sio.loadmat('../data/labels/SYN_radiocarpal_jingshan_QC_MST_2_3_Month.mat')['BMEL_Mask'])
	#type(image)
