import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_img_and_labels(iml, i):
        return iml[i,0,:,:,:], iml[i,1,:,:,:]

def load_preds(i, tag):
    return np.load('../../../../preds/{0}/img{1}_{2}.npy'.format(tag,i,tag))

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
    p = np.load('../../preds/{0}/img{1}d{2}_{3}_preds.npy'.format(tag,i,d,tag))
    
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

def save_pre_post_given_mask(iml, i, mask, savetag): 
    img, img_labels = get_img_and_labels(iml, i, d)
    mask_post = postprocess(mask)
    #overlay_mask_and_save(img, img_labels, '../../pictures/img{0}d{1}_{2}_seg_true'.format(i,d,tag))
    overlay_mask_and_save(img, mask, '../../pictures/img{0}d{1}_{2}_seg_pre'.format(i,d,savetag))
    overlay_mask_and_save(img, mask_post, '../../pictures/img{0}d{1}_{2}_seg_post'.format(i,d,savetag))

def save_true_and_mask(iml, i, mask, savetag):
        img, img_labels = get_img_and_labels(iml, i)

        for d in range(20):
            if (d % 5 == 0):
              print(d)
            overlay_mask_and_save(img[:,:,d], img_labels[:,:,d], '../../../../pictures/{0}/img{1}/img{2}d{3}_{4}_true'.format(savetag, i, i, d, savetag))
            overlay_mask_and_save(img[:,:,d], mask[:,:,d], '../../../../pictures/{0}/img{1}/img{2}d{3}_{4}_pred'.format(savetag, i, i, d, savetag))


#def calc_metrics(true_labs, pred_labs):
#
#    # calculate basics
#    tp = np.sum((true_labs == 1) & (pred_labs == 1))
#    tn = np.sum((true_labs == 0) & (pred_labs == 0))
#    fp = np.sum((true_labs == 0) & (pred_labs == 1))
#    fn = np.sum((true_labs == 1) & (pred_labs == 0))
#    p = tp + fn
#    n = tn + fp
#
#    # calculate metrics
#    acc = (tp + tn)/(p + n)
#    sensitivity = tp/p
#    specificity =  tn/n
#    ppv = tp/(tp+fp)
#    npv = tn/(tn + fn)
#    dsc = 2*tp/(2*tp + fp + fn)
#
#    # make dictionary
#    metrics = {'acc': acc, 'sens': sensitivity, 'spec': specificity, 'ppv': ppv, 'npv': npv, 'dsc': dsc}
#
#    return metrics
#
#def save_true_pre_post_batch(iml, imgs, tag, savedir): 
#
#    for i in imgs:
#        for d in range(20):
#            save_true_pre_post_images(iml, i, d, tag, savedir)

#def save_metrics(iml, imgs, tag):
#
#    for z in imgs:
#
#        print(z)
#
#        with open("img{0}_{1}_metrics.txt".format(z, tag), "w") as text_file:
#            raws, pps = [], []
#            for i in range(20):
#                true_labs = iml[z,1,:,:,i]
#                pred_labs = load_preds(z, i, tag)
#                pred_labs_post = postprocess(pred_labs)
#
#                raw_metrics = calc_metrics(true_labs, pred_labs)
#                pp_metrics = calc_metrics(true_labs, pred_labs_post)
#                raws.append(raw_metrics)
#                pps.append(pp_metrics)
#
#                print("Raw Metrics Depth {0}:".format(i), file=text_file)
#                for m, v in raw_metrics.items():
#                    print("\t{0} : {1}".format(m, v), file=text_file)
#
#                print(" ", file=text_file)
#
#                print("Post-processed Metrics Depth {0}:".format(i), file=text_file)
#                for m, v in pp_metrics.items():
#                    print("\t{0} : {1}".format(m, v), file=text_file)
#
#                print("\n", file=text_file)
#
#            print("Averages\n\n", file=text_file)
#            for k in raws[0]:
#                avg_k = np.mean([raws[i][k] for i in range(len(raws)) if math.isnan(raws[i][k]) is False])
#                print("\tRaw {0}: {1}".format(k, avg_k), file=text_file)
#
#            print("\n", file=text_file)
#            for k in pps[0]:
#                avg_k = np.mean([pps[i][k] for i in range(len(pps)) if math.isnan(pps[i][k]) is False])
#                print("\tPost-processed {0}: {1}".format(k, avg_k), file=text_file)
#
#            print("Middle Averages\n\n", file=text_file)
#
#            for k in raws[0]:
#                avg_k = np.mean([raws[i][k] for i in range(4,15) if math.isnan(raws[i][k]) is False])
#                print("\tRaw {0}: {1}".format(k, avg_k), file=text_file)
#
#            print("\n", file=text_file)
#            for k in pps[0]:
#                avg_k = np.mean([pps[i][k] for i in range(4,15) if math.isnan(pps[i][k]) is False])
#                print("\tPost-processed {0}: {1}".format(k, avg_k), file=text_file)

#def save_metrics_whole(iml, imgs, tag):
#
#    for z in imgs:
#
#        print(z)
#
#        with open("img{0}_{1}_metrics_whole.txt".format(z, tag), "w") as text_file: 
#
#            true_labs = iml[z,1,:,:,:]
#            pred_labs = load_preds_whole(z, tag)
#            pred_labs_post = postprocess(pred_labs)
#
#            true_labs_mid = true_labs[:,:,4:14]
#            pred_labs_mid = pred_labs[:,:,4:14]
#            pred_labs_mid_post = postprocess(pred_labs_mid)
#
#            raw_metrics = calc_metrics(true_labs, pred_labs)
#            pp_metrics = calc_metrics(true_labs, pred_labs_post)
#
#            raw_metrics_mid = calc_metrics(true_labs_mid, pred_labs_mid)
#            pp_metrics_mid = calc_metrics(true_labs_mid, pred_labs_mid_post)
#
#            print("Raw Metrics:", file=text_file)
#            for m, v in raw_metrics.items():
#                print("\t{0} : {1}".format(m, v), file=text_file)
#
#            print(" ", file=text_file)
#
#            print("Post-processed Metrics", file=text_file)
#            for m, v in pp_metrics.items():
#                print("\t{0} : {1}".format(m, v), file=text_file)
#
#            print(" ", file=text_file)
#            
#            print("Raw Metrics Mid (4-14):", file=text_file)
#            for m, v in raw_metrics_mid.items():
#                print("\t{0} : {1}".format(m, v), file=text_file)
#
#            print(" ", file=text_file)
#
#            print("Post-processed Metrics Mid (4-14):", file=text_file)
#            for m, v in pp_metrics_mid.items():
#                print("\t{0} : {1}".format(m, v), file=text_file)

#def test_dsc(pred_labs, labs):
#    intersection = np.sum(np.multiply(pred_labs, labs), axis=(1,2,3))
#    preds_sum = np.sum(pred_labs, axis=(1,2,3))
#    labels_sum = np.sum(labs, axis=(1,2,3))
#    stability = 0.00001
#    numerator = np.multiply(intersection, 2)
#    denominator = np.add(np.add(preds_sum, labels_sum), stability)
#    dice_coeff = np.true_divide(numerator, denominator)
#    dice_coeff_mean = np.mean(dice_coeff)
#    return dice_coeff_mean
#
def dsc(pred_labs, labs):
    intersection = np.sum(np.multiply(pred_labs, labs))
    preds_sum = np.sum(pred_labs)
    labels_sum = np.sum(labs)
    stability = 0.00001
    numerator = np.multiply(intersection, 2)
    denominator = np.add(np.add(preds_sum, labels_sum), stability)
    dice_coeff = np.true_divide(numerator, denominator)
    return dice_coeff

if __name__ == '__main__':

    # load all images and labels
    iml = np.load('../../../../data/datasets/images_and_labels.npy')
    #imgn = 1
    tag = 'fullimg_4_571'
    for imgn in range(8):
        mask = load_preds(imgn, tag)
        save_true_and_mask(iml, imgn, mask, tag)
        print(dsc(mask, iml[imgn,1,:,:,:])) 

    #labels = np.load('labs_test.npy').astype(np.float32)
    #imgs = np.load('imgs_test.npy')
    #test_preds = np.ones((24,128,128,16), dtype=np.float32)

    #test_batch_labs = labels[550:574,:,:,:]
    #print(test_dsc(test_preds, test_batch_labs))
