import numpy as np
import matplotlib.pyplot as plt
import postprocess as pp

def load_logits(i, d, tag):
	logs = np.load('../../preds/img{0}d{1}_{2}.npy'.format(i,d,tag)).reshape((262144, 2))
	return logs

def softmax(logits):
	ex = np.exp(logits)
	sums = np.sum(ex, axis=1).reshape((logits.shape[0], 1))
	return ex/sums

def mask(p, threshold):
	return (p[:,1] > threshold).reshape((512,512))
	
if __name__ == '__main__': 

	# load all images and labels
	iml = np.load('../../data/datasets/images_and_labels.npy')

	# load logits and softmax them
	logs = load_logits(8, 9, '2ch_big_logits')
	p = softmax(logs)

	dices_raw = []
	dices_pp = []
	ppvs_raw = []
	ppvs_pp = []

	labs = iml[8,1,:,:,9]
	thresholds = np.linspace(0.1, 0.90, 20)

	# loop through thresholds and save
	for t in thresholds:
		print("Working on {0}.".format(t))

		m = mask(p, t)
		m_pp = pp.postprocess(m)

		"""
		metrics_raw = pp.calc_metrics(labs, m)
		metrics_pp = pp.calc_metrics(labs, m_pp)

		dices_raw.append(metrics_raw['dsc'])
		dices_pp.append(metrics_pp['dsc'])
		ppvs_raw.append(metrics_raw['ppv'])
		ppvs_pp.append(metrics_pp['ppv'])
		"""

		pp.save_pre_post_given_mask(iml, 8, 9, m, '500t{0}'.format(int(t*10000)))

	"""
	plt.plot(thresholds, ppvs_raw, label='raw ppv', color='r')
	plt.plot(thresholds, ppvs_pp, label='postprocessed ppv', color='m')
	plt.plot(thresholds, dices_raw, label='raw dsc', color='c')
	plt.plot(thresholds, dices_pp, label='postprocessed dsc', color= 'g')
	plt.xlabel('Threshold')
	plt.ylabel('Values')
	plt.title('PPV/DSC vs threshold for Image 8 Depth 9')
	plt.legend(loc='best')
	plt.show()
	"""