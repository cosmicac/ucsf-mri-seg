from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import raseg_model
import raseg_input
import util.make_datasets_fullconv as md

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../../../models/raseg_predict',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../models/raseg_train_fullconv_bme256',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('imgn', 0, """Image number to evaluate.""")
tf.app.flags.DEFINE_string('savetag', 'fullconv_bme256_3900', """Tag to save predictions as. """)

"""Assumes image is of dimensions [height, width, depth, nchannels]"""
def normalize(image):
  h, w, d = raseg_input.PATCH_HEIGHT, raseg_input.PATCH_WIDTH, raseg_input.PATCH_DEPTH
  mean = np.mean(image, axis=(0,1,2))
  var = np.var(image, axis=(0,1,2))
  adjusted_stddev = np.maximum(np.sqrt(var), np.divide(1.0,np.sqrt(h*w*d)))
  normalized_images = np.divide(np.subtract(image, mean), adjusted_stddev)
  return normalized_images

def predict():

  # load images, both channels
  images_and_labels = np.load('../../../data/datasets/t2imgs_and_prereg_labels.npy')

  img = images_and_labels[FLAGS.imgn,0,:,:,:]
  labs = images_and_labels[FLAGS.imgn,1,:,:,:]

  # Indexes for the 32 patches to predict
  patches = np.zeros((4,256,256,20,1), dtype=np.float32)
  depth_idx = [10]
  hw_idx = [128, 384]

  # extract patches for classification
  k = 0
  for d in depth_idx:
    for h in hw_idx:
      for w in hw_idx:
        patches[k,:,:,:,0] = md.extract_patch(img, (h,w,d)).astype(np.float32)
        k += 1

  # Normalize patches and make batches. 
  for i in range(patches.shape[0]):
    patches[i,:,:,:,:] = normalize(patches[i,:,:,:,:])
  batches = [patches[0:2,:,:,:,:], patches[2:4,:,:,:,:]]  
  preds = np.zeros((4,256,256,20))

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
                
  # Stitch together for the final mask. 
  final_mask = np.zeros((512,512,20))
  c = 0
  for d in depth_idx:
    for h in hw_idx:
      for w in hw_idx:
        final_mask[h-128:h+128,w-128:w+128,d-10:d+10] = preds[c]
        c += 1

  # Save the mask.
  np.save('../../../preds/{0}/img{1}_{2}'.format(FLAGS.savetag, FLAGS.imgn, FLAGS.savetag), final_mask)

  return final_mask

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  predict()

if __name__ == '__main__':
  tf.app.run()
