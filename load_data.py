import numpy as np 
import scipy.io as sio
import matplotlib.image as img 
import matplotlib.pyplot as plt

if __name__ == '__main__':

	"""
	qc_mat_contents = sio.loadmat('data/labels/SYN_radiocarpal_jingshan_QC_MST_6_Baseline.mat')
	og_mat_contents = sio.loadmat('data/labels/SYN_radiocarpal_jingshan_6_Baseline.mat')
	qc_mask = np.asarray(qc_mat_contents['BMEL_Mask'])
	og_mask = np.asarray(og_mat_contents['BMEL_Mask'])
	print(np.shape(qc_mask))
	print(np.shape(og_mask))
	plt.imshow(og_mask[:, :, 7], cmap = 'gray', interpolation='none')
	plt.imshow(qc_mask[:, :, 7], cmap = 'bwr', alpha=0.5, interpolation = 'none')
	#plt.imshow(image[:, :, 9], cmap = 'bwr')	
	plt.show()
	"""

	image = sio.loadmat('../data/raw_images/post_2_3_Month.mat')
	type(image)
