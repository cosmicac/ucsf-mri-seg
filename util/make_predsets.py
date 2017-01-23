import numpy as np
import make_datasets
import npy_to_bin 

def imgslice_to_patch_arr(imgc1, imgc2, depth):
	
	h, w, d = imgc1.shape
	patch_arr = []
	count = 0

	for i in range(h):
		for j in range(w):

			if count % 25000 == 0:
				print("Extracting patch {0}.".format(count)) 

			patch_ijc1 = make_datasets.extract_patch(imgc1, center=(i,j,depth))
			patch_ijc2 = make_datasets.extract_patch(imgc2, center=(i,j,depth))

			# concatenate the patches
			patch_ij = np.concatenate((patch_ijc1[...,np.newaxis], patch_ijc2[...,np.newaxis]), axis=3)
			patch_arr.append(patch_ij)
			count += 1

	return np.array(patch_arr) 

def predict_slice(images_and_labels, pre_images, img_number, depth, save_path):
	c1 = images_and_labels[img_number, 0, :, :, :]
	c2 = pre_images[img_number]
	img_labels = images_and_labels[img_number, 1, :, :, depth].astype('uint16').flatten()
	img = imgslice_to_patch_arr(c1, c2, depth).astype('uint16')
	npy_to_bin.flatten_and_bin(img, img_labels, save_path)


if __name__ == '__main__':

	# load images and labels
	images_and_labels = np.load('../../data/datasets/images_and_labels.npy')
	pre_images = np.load('../../data/datasets/pre_images.npy')
	n = len(images_and_labels)
	print(n)

	"""	
	img8c1 = images_and_labels[8,0,:,:,:]
	img8c2 = pre_images[8]
	img8d10_labels = images_and_labels[8,1,:,:,10].astype('uint16').flatten()
	img8d10 = imgslice_to_patch_arr(img8c1, img8c2, 10).astype('uint16')

	npy_to_bin.flatten_and_bin(img8d11, img8d11_labels, '../../data/datasets/bins/img8d10_and_label_2ch_batch_1.bin')
	"""
	
	predict_slice(images_and_labels, pre_images, 8, 10, '../../data/datasets/bins/img8d10_and_label_2ch_batch_1.bin')
	predict_slice(images_and_labels, pre_images, 8, 11, '../../data/datasets/bins/img8d11_and_label_2ch_batch_1.bin')
	

	"""
	img21c1 = images_and_labels[21,0,:,:,:]
	img21c2 = pre_images[21]
	img21d7_labels = images_and_labels[21,1,:,:,7].astype('uint16').flatten()
	img21d7 = imgslice_to_patch_arr(img21c1, img21c2, 7).astype('uint16')

	img27c1 = images_and_labels[27,0,:,:,:]
	img27c2 = pre_images[27]
	img27d9_labels = images_and_labels[27,1,:,:,9].astype('uint16').flatten()
	img27d9 = imgslice_to_patch_arr(img27c1, img27c2, 9).astype('uint16')

	train = np.concatenate((img21d7, img27d9), axis=0)
	train_labels = np.concatenate((img21d7_labels, img27d9_labels), axis=0)

	# shuffle 
	p = np.random.permutation(512*512*2)
	train = train[p]
	labels = train_labels[p]

	print(train.shape)
	print(train_labels.shape)

	# todo don't hardcode this
	div = (512*512*2)/8
	train1 = train[:div,:,:,:,:]
	train2 = train[div:2*div,:,:,:,:]
	train3 = train[2*div:3*div,:,:,:,:]
	train4 = train[3*div:4*div,:,:,:,:]
	train5 = train[4*div:5*div,:,:,:,:]
	train6 = train[5*div:6*div,:,:,:,:]
	train7 = train[6*div:7*div,:,:,:,:]
	train8 = train[7*div:8*div,:,:,:,:]
	
	labels1 = labels[:div]
	labels2 = labels[div:2*div]
	labels3 = labels[2*div:3*div]
	labels4 = labels[3*div:4*div]
	labels5 = labels[4*div:5*div]
	labels6 = labels[5*div:6*div]
	labels7 = labels[6*div:7*div]
	labels8 = labels[7*div:8*div]	

	npy_to_bin.flatten_and_bin(train1, labels1, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_1.bin')
	npy_to_bin.flatten_and_bin(train2, labels2, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_2.bin')
	npy_to_bin.flatten_and_bin(train3, labels3, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_3.bin')
	npy_to_bin.flatten_and_bin(train4, labels4, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_4.bin')
	npy_to_bin.flatten_and_bin(train5, labels5, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_5.bin')
	npy_to_bin.flatten_and_bin(train6, labels6, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_6.bin')
	npy_to_bin.flatten_and_bin(train7, labels7, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_7.bin')
	npy_to_bin.flatten_and_bin(train8, labels8, '../../data/datasets/bins/train_and_label_2ch_imgs_batch_8.bin')
	"""