import numpy as np
import matplotlib.pyplot as plt
import postprocess as pp
import os


def view_images(iml, ids, savepath):
	for i,d in ids:
		img, img_labels = pp.get_img_and_labels(iml, i, d)
		pp.overlay_mask_and_save(img, img_labels, '{0}/img{1}d{2}'.format(savepath,i,d))



if __name__ == '__main__':

	# load all images and labels
	iml = np.load('../../data/datasets/images_and_labels.npy')

	for i in range(20,32):
		os.makedirs('../../pictures/raw_pics/img{0}'.format(i))
		img = iml[i,0,:,:,:]
		for d in range(20):
			plt.imshow(img[:,:,d], cmap='bone')
			plt.savefig('../../pictures/raw_pics/img{0}/img{0}d{1}'.format(i, d), format='png')
			print('Saved: ../../pictures/raw_pics/img{0}/img{0}d{1}'.format(i, d))


	"""
	ids = []
	for i in [4,9,14,18,22,27,32]:
		ids.extend(zip(np.repeat(i,8), np.arange(7,15)))

	view_images(iml, ids, '../../pictures/view')
	"""