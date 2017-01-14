import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_img_and_labels(iml, i, d):
	return iml[i,0,:,:,d], iml[i,1,:,:,d]

def load_preds(i, d):
	return np.load('../../preds/img{0}d{1}_preds.npy'.format(i,d)).reshape((512,512))

def overlay_mask_and_save(img, mask, filename):
	plt.imshow(img, cmap='bone')
	plt.imshow(mask, cmap='bwr', interpolation='None', alpha=0.2)
	plt.savefig(filename, format='png')

def postprocess(img):
	opened = ndimage.binary_opening(img)
	closed = ndimage.binary_closing(opened)
	open_closed_med = ndimage.median_filter(closed, 5)
	return open_closed_med

def save_true_pre_post_images(iml, i, d):
	img, img_labels = get_img_and_labels(iml, i, d)
	p = np.load('../../preds/img{0}d{1}_preds.npy'.format(i,d)).reshape((512, 512))
	p_post = postprocess(p)

	overlay_mask_and_save(img, img_labels, '../../pictures/img{0}d{1}_seg_true'.format(i, d))
	overlay_mask_and_save(img, p, '../../pictures/img{0}d{1}_seg_pre'.format(i,d))
	overlay_mask_and_save(img, p_post, '../../pictures/img{0}d{1}_seg_post'.format(i,d))

def calc_metrics(true_labs, pred_labs):

	# calculate basics
	tp = np.sum((true_labs == 1) & (pred_labs == 1))
	tn = np.sum((true_labs == 0) & (pred_labs == 0))
	fp = np.sum((true_labs == 0) & (pred_labs == 1))
	fn = np.sum((true_labs == 1) & (pred_labs == 0))
	p = tp + fn
	n = tn + fp

	# calculate metrics
	acc = (tp + tn)/(p + n)
	sensitivity = tp/p
	specificity =  tn/n
	ppv = tp/(tp+fp)
	npv = tn/(tn + fn)
	dsc = 2*tp/(2*tp + fp + fn)

	# make dictionary
	metrics = {'acc': acc, 'sens': sensitivity, 'spec': specificity, 'ppv': ppv, 'npv': npv, 'dsc': dsc}

	return metrics

if __name__ == '__main__':

	# load all images and labels
	iml = np.load('../../data/datasets/images_and_labels.npy')

	save_true_pre_post_images(iml, 8, 11)

	"""	
	true_labs9 = iml[8,1,:,:,9]
	pred_labs9 = load_preds(8, 9)
	print("Depth 9")
	print(calc_metrics(true_labs9, pred_labs9))

	
	true_labs10 = iml[8,1,:,:,10]
	pred_labs10 = load_preds(8, 10)
	print("Depth 10")
	print(calc_metrics(true_labs10, pred_labs10))

	true_labs11 = iml[8,1,:,:,11]
	pred_labs11 = load_preds(8, 11)
	print("Depth 11")
	print(calc_metrics(true_labs11, pred_labs11))

	pred_labs9_post = postprocess(pred_labs9)
	pred_labs10_post = postprocess(pred_labs10)
	pred_labs11_post = postprocess(pred_labs11)

	print("Depth 9 post")
	print(calc_metrics(true_labs9, pred_labs9_post))

	print("Depth 10 post")
	print(calc_metrics(true_labs10, pred_labs10_post))

	print("Depth 11 post")
	print(calc_metrics(true_labs11, pred_labs11_post))
	"""
