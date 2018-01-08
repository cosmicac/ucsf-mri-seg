from __future__ import division
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import raseg_model
import raseg_input

parser = argparse.ArgumentParser(description='Evaluates a raseg model.')
parser.add_argument('-t', '--tag', help='Tag to identify the dataset.', required=True)
args = vars(parser.parse_args())
TAG = args['tag']

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'models/raseg_predict_{0}'.format(TAG),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'models/raseg_train_{0}'.format(TAG),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('imgn', 0, """Image number to evaluate.""")

"""Assumes image is of dimensions [height, width, depth, nchannels]"""
def normalize(image):
  h, w, d = raseg_input.PATCH_HEIGHT, raseg_input.PATCH_WIDTH, raseg_input.PATCH_DEPTH
  mean = np.mean(image, axis=(0,1,2))
  var = np.var(image, axis=(0,1,2))
  adjusted_stddev = np.maximum(np.sqrt(var), np.divide(1.0,np.sqrt(h*w*d)))
  normalized_images = np.divide(np.subtract(image, mean), adjusted_stddev)
  return normalized_images

def make_dir_if_needed(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def overlay_mask_and_save(img, mask, filename):
    plt.imshow(img, cmap='bone')
    plt.imshow(mask, cmap='brg', interpolation='None', alpha=0.2)
    plt.savefig(filename, format='png')

def dsc(pred_labs, labs):
    intersection = np.sum(np.multiply(pred_labs, labs))
    preds_sum = np.sum(pred_labs)
    labels_sum = np.sum(labs)
    stability = 0.00001
    numerator = np.multiply(intersection, 2)
    denominator = np.add(np.add(preds_sum, labels_sum), stability)
    dice_coeff = np.true_divide(numerator, denominator)
    return dice_coeff

def save_true_and_mask(iml, i, mask, savetag):
        img, img_labels = iml[i,0,:,:,:], iml[i,1,:,:,:]

        # Create directory to save pictures in if neccessary.
        directory = 'pictures/{0}/img{1}'.format(savetag, i)
        make_dir_if_needed(directory)

        for d in range(20):
            if (d % 5 == 0):
              print(d)
            overlay_mask_and_save(img[:,:,d], img_labels[:,:,d], '../../../../pictures/{0}/img{1}/img{2}d{3}_{4}_true'.format(savetag, i, i, d, savetag))
            overlay_mask_and_save(img[:,:,d], mask[:,:,d], '../../../../pictures/{0}/img{1}/img{2}d{3}_{4}_pred'.format(savetag, i, i, d, savetag))

def predict():

    # load images, both channels
    images_and_labels = np.load('datasets/images_and_labels_{0}.npy'.format(TAG))
    pre_images = np.load('datasets/pre_images_{0}.npy'.format(TAG))
  
    img = images_and_labels[FLAGS.imgn,0,:,:,:]
    pre_img = pre_images[FLAGS.imgn,:,:,:]
    labs = images_and_labels[FLAGS.imgn,1,:,:,:]
  
    patch = np.concatenate((img[...,np.newaxis], pre_img[...,np.newaxis]), axis=3)
    batches = [normalize(patch).reshape((1,512,512,20,2))]
    preds = np.zeros((1,512,512,20))
    with tf.Graph().as_default() as g:
  
      # Build graph and prediction operation.
      images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,raseg_input.PATCH_WIDTH,raseg_input.PATCH_WIDTH, raseg_input.PATCH_DEPTH,raseg_input.NCHANNELS))
      logits = raseg_model.inference(images)
      softmax = tf.nn.softmax(logits, dim=-1)
      preds_op = tf.argmax(softmax, axis=4)
   
      # Restore the moving average version of the learaned variables for prediction.
      variable_averages = tf.train.ExponentialMovingAverage(
          raseg_model.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
  
      with tf.Session() as sess:
  
        # Load checkpoint state.
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
          print('No checkpoint file found.')
          return
  
        c = 0
        for b in batches:
          
          # Run the session, feeding each batch in and fill in predictions.
          preds_batch = sess.run(preds_op, feed_dict={images: b})
          preds[FLAGS.batch_size*c:FLAGS.batch_size*(c+1),:,:,:] = preds_batch
          c += 1
                  
    final_mask = preds.reshape((512,512,20))
    make_dir_if_needed('preds/{0}'.format(TAG))
    np.save('preds/{0}/img{1}_{2}'.format(TAG, FLAGS.imgn, TAG), final_mask)
    save_true_and_mask(images_and_labels, FLAGs.imgn, final_mask, TAG)
    return final_mask

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  predict()

if __name__ == '__main__':
  tf.app.run()
