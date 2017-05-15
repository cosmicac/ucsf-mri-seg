import numpy as np 
import scipy.io as sio
import os.path

TIMES = ['Baseline', '1_Month', '3_Month', '1_Year']
IMAGE_PATH_TEMPLATE = '../data/raw_images/post_{0}_{1}.mat'
LABEL_PATH_TEMPLATE = '../data/merged_labels/{0}_{1}.npy'

if __name__ == '__main__':

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