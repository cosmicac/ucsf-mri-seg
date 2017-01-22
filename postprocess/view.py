import numpy as np
import matplotlib.pyplot as plt
import postprocess as pp


def view_images(iml, ids, savepath):
	for i,d in ids:
		img, img_labels = pp.get_img_and_labels(iml, i, d)
		pp.overlay_mask_and_save(img, img_labels, '{0}/img{1}d{2}'.format(savepath,i,d))



if __name__ == '__main__':

	# load all images and labels
	iml = np.load('../../data/datasets/images_and_labels.npy')

	ids = []
	for i in [4,9,14,18,22,27,32]:
		ids.extend(zip(np.repeat(i,8), np.arange(7,15)))

	view_images(iml, ids, '../../pictures/view')
	