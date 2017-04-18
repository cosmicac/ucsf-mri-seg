import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_img_and_labels(iml, i, d):
	return iml[i,0,:,:,d], iml[i,1,:,:,d]

def load_preds(i, d, tag):
	return np.load('../../preds/img{0}d{1}_{2}_preds.npy'.format(i,d,tag)).reshape((512,512))

def load_preds_whole(i, tag):
	preds = np.zeros((512,512,20))
	for d in range(20):
		preds[:,:,d] = np.load('../../preds/img{0}d{1}_{2}_preds.npy'.format(i,d,tag)).reshape((512,512))
	return preds

def overlay_mask_and_save(img, mask, filename):
	plt.imshow(img, cmap='bone')
	plt.imshow(mask, cmap='brg', interpolation='None', alpha=0.2)
	plt.savefig(filename, format='png')

def postprocess(img):
	opened = ndimage.binary_opening(img)
	closed = ndimage.binary_closing(opened)
	open_closed_med = ndimage.median_filter(closed, 5)
	return open_closed_med

def save_true_pre_post_images(iml, i, d, tag, dir=None):
	img, img_labels = get_img_and_labels(iml, i, d)
	p = np.load('../../preds/img{0}d{1}_{2}_preds.npy'.format(i,d,tag))
	
	if p.shape != img.shape:
		p = p.reshape(img.shape)

	p_post = postprocess(p)

	if dir:
		overlay_mask_and_save(img, img_labels, '{0}/img{1}d{2}_{3}_seg_true'.format(dir, i,d,tag))
		overlay_mask_and_save(img, p, '{0}/img{1}d{2}_{3}_seg_pre'.format(dir, i,d,tag))
		overlay_mask_and_save(img, p_post, '{0}/img{1}d{2}_{3}_seg_post'.format(dir, i,d,tag))
	else:
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

def save_true_pre_post_batch(iml, imgs, tag, savedir): 

	for i in imgs:
		for d in range(20):
			save_true_pre_post_images(iml, i, d, tag, savedir)

def save_metrics(iml, imgs, tag):

	for z in imgs:

		print(z)

		with open("img{0}_{1}_metrics.txt".format(z, tag), "w") as text_file:
			raws, pps = [], []
			for i in range(20):
				true_labs = iml[z,1,:,:,i]
				pred_labs = load_preds(z, i, tag)
				pred_labs_post = postprocess(pred_labs)

				raw_metrics = calc_metrics(true_labs, pred_labs)
				pp_metrics = calc_metrics(true_labs, pred_labs_post)
				raws.append(raw_metrics)
				pps.append(pp_metrics)

				print("Raw Metrics Depth {0}:".format(i), file=text_file)
				for m, v in raw_metrics.items():
					print("\t{0} : {1}".format(m, v), file=text_file)

				print(" ", file=text_file)

				print("Post-processed Metrics Depth {0}:".format(i), file=text_file)
				for m, v in pp_metrics.items():
					print("\t{0} : {1}".format(m, v), file=text_file)

				print("\n", file=text_file)

			print("Averages\n\n", file=text_file)
			for k in raws[0]:
				avg_k = np.mean([raws[i][k] for i in range(len(raws)) if math.isnan(raws[i][k]) is False])
				print("\tRaw {0}: {1}".format(k, avg_k), file=text_file)

			print("\n", file=text_file)
			for k in pps[0]:
				avg_k = np.mean([pps[i][k] for i in range(len(pps)) if math.isnan(pps[i][k]) is False])
				print("\tPost-processed {0}: {1}".format(k, avg_k), file=text_file)

			print("Middle Averages\n\n", file=text_file)

			for k in raws[0]:
				avg_k = np.mean([raws[i][k] for i in range(4,15) if math.isnan(raws[i][k]) is False])
				print("\tRaw {0}: {1}".format(k, avg_k), file=text_file)

			print("\n", file=text_file)
			for k in pps[0]:
				avg_k = np.mean([pps[i][k] for i in range(4,15) if math.isnan(pps[i][k]) is False])
				print("\tPost-processed {0}: {1}".format(k, avg_k), file=text_file)

def save_metrics_whole(iml, imgs, tag):

	for z in imgs:

		print(z)

		with open("img{0}_{1}_metrics_whole.txt".format(z, tag), "w") as text_file:	

			true_labs = iml[z,1,:,:,:]
			pred_labs = load_preds_whole(z, tag)
			pred_labs_post = postprocess(pred_labs)

			true_labs_mid = true_labs[:,:,4:14]
			pred_labs_mid = pred_labs[:,:,4:14]
			pred_labs_mid_post = postprocess(pred_labs_mid)

			raw_metrics = calc_metrics(true_labs, pred_labs)
			pp_metrics = calc_metrics(true_labs, pred_labs_post)

			raw_metrics_mid = calc_metrics(true_labs_mid, pred_labs_mid)
			pp_metrics_mid = calc_metrics(true_labs_mid, pred_labs_mid_post)

			print("Raw Metrics:", file=text_file)
			for m, v in raw_metrics.items():
				print("\t{0} : {1}".format(m, v), file=text_file)

			print(" ", file=text_file)

			print("Post-processed Metrics", file=text_file)
			for m, v in pp_metrics.items():
				print("\t{0} : {1}".format(m, v), file=text_file)

			print(" ", file=text_file)
			
			print("Raw Metrics Mid (4-14):", file=text_file)
			for m, v in raw_metrics_mid.items():
				print("\t{0} : {1}".format(m, v), file=text_file)

			print(" ", file=text_file)

			print("Post-processed Metrics Mid (4-14):", file=text_file)
			for m, v in pp_metrics_mid.items():
				print("\t{0} : {1}".format(m, v), file=text_file)


if __name__ == '__main__':

	# load all images and labels
	iml = np.load('../../data/datasets/t2imgs_and_prereg_labels.npy')

	# t2 val
	val_imgs = np.concatenate((np.arange(6), np.arange(29,31)))

	save_metrics_whole(iml, val_imgs, 't2bmeonly')

	# save_true_pre_post_batch(iml, val_imgs, 't2bmeonly', '../../pictures/t2bmeonly_validation/')

	# save_true_pre_post_images(iml, 3, 11, 't2bmeonly')
	
	"""
	true_labs = iml[8,1,:,:,11]
	pred_labs = load_preds(5, 8, 'regfix')
	pp_labs = postprocess(pred_labs)
	print(calc_metrics(true_labs, pred_labs))
	print(calc_metrics(true_labs,pp_labs))
	"""


	"""
	for i in range(20):
		save_true_pre_post_images(iml, 8, i, 'kmeans_partial')
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