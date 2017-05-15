import numpy as np 
import os.path
import scipy.io as sio
import sys
import matplotlib.pyplot as plt

TIMES = ['Baseline', '1_Month', '3_Month', '1_Year']
IMAGE_PATH_TEMPLATE = '../data/raw_images/post_{0}_{1}.mat'
LABEL_PATH_TEMPLATE = '../data/labels/{0}_{1}_{2}.mat'

POSSIBLE_RADIOCARPAL = ['SYN_radiocarpal_jingshan_QC_MST',
					   'SYN_radiocarpal_jingshan',
					   'SYN_radio-carpal_joint_jingshan']

POSSIBLE_RADIOULNAR = ['SYN_radioulnar_jingshan_QC_MST',
					   'SYN_radioulnar_jingshan',
					   'SYN_distal_radio-ulnar_joint_jingshan']

POSSIBLE_INTERCARPAL = ['SYN_intercarpal-CMCJ_jingshan_QC_MST',
						'SYN_intercarpal-CMCJ_jingshan',
						'SYN_intercarpal_jingshan']

POSSIBLE_TENDON = ['SYN_tendon_jingshan_QC_MST',
				   'SYN_tendon_jingshan']

if __name__ == '__main__':

	# loop over time points and base paths
	for i in range(1, 18):
		for time in TIMES:

			# construct path for raw image file
			raw_image_file = IMAGE_PATH_TEMPLATE.format(i, time)

			# only merge labels if the raw image file exists
			if os.path.isfile(raw_image_file):
				print('Merging for patient {0} at time {1}.'.format(i, time))

				# load image to get shape
				raw_image = sio.loadmat(raw_image_file)['ImageWTR']

				merged_mask = np.zeros(raw_image.shape, dtype=bool)
				nonempty_labels = False

				# merge with radiocarpal masks, if exist
				# loop in preferable order with QC'd as most favorable
				for rc in POSSIBLE_RADIOCARPAL:
					label_file = LABEL_PATH_TEMPLATE.format(rc, i, time)
					# only merge the labels if they exist
					if os.path.isfile(label_file):
						rc_label = sio.loadmat(label_file)['BMEL_Mask']
						rc_label_mask = (rc_label == 1)
						merged_mask = np.logical_or(merged_mask, rc_label_mask)
						nonempty_labels = True
						break

				# merge with radioulnar masks, if exist
				# loop in preferable order with QC'd as most favorable
				for ru in POSSIBLE_RADIOULNAR:
					label_file = LABEL_PATH_TEMPLATE.format(ru, i, time)
					# only merge the labels if they exist
					if os.path.isfile(label_file):
						ru_label = sio.loadmat(label_file)['BMEL_Mask']
						ru_label_mask = (ru_label == 1)
						merged_mask = np.logical_or(merged_mask, ru_label_mask)
						nonempty_labels = True
						break

				# merge with intercarpal masks, if exist
				# loop in preferable order with QC'd as most favorable
				for ic in POSSIBLE_INTERCARPAL:
					label_file = LABEL_PATH_TEMPLATE.format(ic, i, time)
					# only merge the labels if they exist
					if os.path.isfile(label_file):
						ic_label = sio.loadmat(label_file)['BMEL_Mask']
						ic_label_mask = (ic_label == 1)
						merged_mask = np.logical_or(merged_mask, ic_label_mask)
						nonempty_labels = True
						break

				# merge with tendon masks, if exist
				# loop in preferable order with QC'd as most favorable
				for te in POSSIBLE_TENDON:
					label_file = LABEL_PATH_TEMPLATE.format(te, i, time)
					# only merge the labels if they exist
					if os.path.isfile(label_file):
						te_label = sio.loadmat(label_file)['BMEL_Mask']
						te_label_mask = (te_label == 1)
						merged_mask = np.logical_or(merged_mask, te_label_mask)
						nonempty_labels = True
						break

				# only save if at least one label was merged
				if nonempty_labels:
					labels = merged_mask.astype(int)
					np.save('../data/merged_labels/{0}_{1}'.format(i, time), labels)




