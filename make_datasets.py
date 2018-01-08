import argparse
import numpy as np 

parser = argparse.ArgumentParser(description='Takes in datasets in npy format and outputs binaries to train on.')
parser.add_argument('-t','--tag', help='Tag to identify the dataset.', required=True)
parser.add_argument('-v','--val', help='Desired size of the validation set.', required=True)
args = vars(parser.parse_args())

def flatten_and_bin(imgs, labels, save_path):
    n, h, w, d, nc = imgs.shape

    # flatten images and labels
    imgs = imgs.reshape((n, h*w*d*nc))
    labels = labels.reshape((n, h*w*d))

    # combine stack them horizontally
    # every row is flattened labels, then the flattened image
    imgs_and_labels_flat = np.hstack((labels, imgs))

    print("Saving to {0}".format(save_path))
    imgs_and_labels_flat.tofile(save_path)

def make_fc_fullimg_dataset(tag, val_size):

    # load images and labels
    images_and_labels = np.load('datasets/images_and_labels_{0}.npy'.format(tag))
    pre_images = np.load('datasets/pre_images_{0}.npy'.format(tag))

    assert pre_images.shape[0] == images_and_labels.shape[0]

    # validation set
    n_before_val = pre_image.shape[0]
    ids = np.arange(n_before_val-val_size)
    images_and_labels = images_and_labels[ids,:,:,:,:]
    pre_images = pre_images[ids,:,:,:]
    n = pre_images.shape[0]
 
    c1, c2 = images_and_labels[:,0,:,:,:], pre_images
    imgs = np.concatenate((c1[...,np.newaxis], c2[...,np.newaxis]), axis=4).astype('uint16')
    labels = images_and_labels[:,1,:,:,:].astype('uint16')

    print(imgs.shape)
    print(labels.shape)
    print(imgs.dtype)
    print(labels.dtype)

    # Split into 5 bins.
    bs = n//5

    imgs1 = imgs[:bs,:,:,:,:]
    imgs2 = imgs[bs:2*bs,:,:,:,:]
    imgs3 = imgs[2*bs:3*bs,:,:,:,:]
    imgs4 = imgs[3*bs:4*bs,:,:,:,:]
    imgs5 = imgs[4*bs:,:,:,:,:]
    
    labels1 = labels[:bs,:,:,:]
    labels2 = labels[bs:2*bs,:,:,:]
    labels3 = labels[2*bs:3*bs,:,:,:]
    labels4 = labels[3*bs:4*bs,:,:,:]
    labels5 = labels[4*bs:,:,:,:]
    
    flatten_and_bin(imgs1, labels1, 'datasets/bins/train_and_label_{0}_batch_1.bin'.format(tag))
    flatten_and_bin(imgs2, labels2, 'datasets/bins/train_and_label_{0}_batch_2.bin'.format(tag))
    flatten_and_bin(imgs3, labels3, 'datasets/bins/train_and_label_{0}_batch_3.bin'.format(tag))
    flatten_and_bin(imgs4, labels4, 'datasets/bins/train_and_label_{0}_batch_4.bin'.format(tag))
    flatten_and_bin(imgs5, labels5, 'datasets/bins/train_and_label_{0}_batch_5.bin'.format(tag))

if __name__ == '__main__':
    tag, val_size = args['tag'], args['val_size']
    make_fc_fullimg_expanded_dataset(tag, val_size)
	
