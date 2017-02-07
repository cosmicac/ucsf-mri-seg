import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_img_and_labels(iml, i, d):
	return iml[i,0,:,:,d], iml[i,1,:,:,d]

def load_preds(i, d, tag):
	return np.load('../../preds/img{0}d{1}_{2}_preds.npy'.format(i,d,tag)).reshape((512,512))

def overlay_mask_and_save(img, mask, filename):
	plt.imshow(img, cmap='bone')
	plt.imshow(mask, cmap='bwr', interpolation='None', alpha=0.2)
	plt.savefig(filename, format='png')

def postprocess(img):
	opened = ndimage.binary_opening(img)
	closed = ndimage.binary_closing(opened)
	open_closed_med = ndimage.median_filter(closed, 5)
	return open_closed_med

def save_true_pre_post_images(iml, i, d, tag):
	img, img_labels = get_img_and_labels(iml, i, d)
	p = np.load('../../preds/img{0}d{1}_{2}_preds.npy'.format(i,d,tag))
	
	if p.shape != img.shape:
		p = p.reshape(img.shape)

	p_post = postprocess(p)
	overlay_mask_and_save(img, img_labels, '../../pictures/img{0}d{1}_{2}_seg_true'.format(i,d,tag))
	overlay_mask_and_save(img, p, '../../pictures/img{0}d{1}_{2}_seg_pre'.format(i,d,tag))
	overlay_mask_and_save(img, p_post, '../../pictures/img{0}d{1}_{2}_seg_post'.format(i,d,tag))

def save_pre_post_given_mask(iml, i, d, mask, savetag): 
	img, img_labels = get_img_and_labels(iml, i, d)
	mask_post = postprocess(mask)
	#overlay_mask_and_save(img, img_labels, '../../pictures/img{0}d{1}_{2}_seg_true'.format(i,d,tag))
	overlay_mask_and_save(img, mask, '../../pictures/img{0}d{1}_{2}_seg_pre'.format(i,d,savetag))
	overlay_mask_and_save(img, mask_post, '../../pictures/img{0}d{1}_{2}_seg_post'.format(i,d,savetag))

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


	save_true_pre_post_images(iml, 5, 8, 'kmeans_partial')

	"""
	for i in range(20):
		save_true_pre_post_images(iml, 8, i, 'kmeans_partial')
	
	with open("img8_kmeans_partial_metrics.txt", "w") as text_file:
		for i in range(20):
			true_labs = iml[8,1,:,:,i]
			pred_labs = load_preds(8, i, 'kmeans')
			pred_labs_post = postprocess(pred_labs)

			raw_metrics = calc_metrics(true_labs, pred_labs)
			pp_metrics = calc_metrics(true_labs, pred_labs_post)

			print("Raw Metrics Depth {0}:".format(i), file=text_file)
			for m, v in raw_metrics.items():
				print("\t{0} : {1}".format(m, v), file=text_file)

			print(" ", file=text_file)

			print("Post-processed Metrics Depth {0}:".format(i), file=text_file)
			for m, v in pp_metrics.items():
				print("\t{0} : {1}".format(m, v), file=text_file)

			print("\n", file=text_file)
	"""
	
	"""
	true_labs9 = iml[8,1,:,:,9]
	pred_labs9 = load_preds(8, 9, '2ch_big')

	true_labs10 = iml[8,1,:,:,10]
	pred_labs10 = load_preds(8, 10, '2ch_big')

	true_labs11 = iml[8,1,:,:,11]
	pred_labs11 = load_preds(8, 11, '2ch_big')

	print("Depth 9")
	print(calc_metrics(true_labs9, pred_labs9))

	print("Depth 10")
	print(calc_metrics(true_labs10, pred_labs10))

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