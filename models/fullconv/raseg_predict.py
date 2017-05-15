from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import raseg_model
import util.make_datasets as md

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../../../models/raseg_predict',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../models/raseg_train_fullconv_3',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('imgn', 0, """Image number to evaluate.""")

def predict():

  # load images, both channels
  images_and_labels = np.load('../../../data/datasets/images_and_labels.npy')
  pre_images = np.load('../../../data/datasets/pre_images.npy')

  img = images_and_labels[FLAGS.imgn,0,:,:,:]
  pre_img = pre_images[FLAGS.imgn,:,:,:]
  labs = images_and_labels[FLAGS.imgn,1,:,:,:]

  # Indexes for the 32 patches to predict
  patches = np.zeros((32,128,128,16,2), dtype=np.float32)
  depth_idx = [8,12]
  hw_idx = [64, 192, 320, 448]

  # extract patches for classification
  k = 0
  for d in depth_idx:
    for h in hw_idx:
      for w in hw_idx:
        patches[k,:,:,:,0] = md.extract_patch(img, (h,w,d)).astype(np.float32)
        patches[k,:,:,:,1] = md.extract_patch(pre_img, (h,w,d)).astype(np.float32)
        k += 1

  batches = [patches[:8,:,:,:,:], patches[8:16,:,:,:,:], patches[16:24,:,:,:,:], patches[24:,:,:,:,:]]  
  preds = np.zeros((32,128,128,16))

  with tf.Graph().as_default() as g:

    # Build graph and prediction operation.
    images = tf.placeholder(tf.float32, shape=(8,128,128,16,2))
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
        preds[8*c:8*(c+1),:,:,:] = preds_batch
        c += 1

  # Stitch together for the final mask. 
  final_mask = np.zeros((512,512,20))
  c = 0
  for d in depth_idx:
    for h in hw_idx:
      for w in hw_idx:
        if c < 16:
          final_mask[h-64:h+64,w-64:w+64,d-8:d+8] = preds[c]
        else:
          final_mask[h-64:h+64,w-64:w+64,16:] = preds[c,:,:,12:]
        c += 1

  # Save the mask.
  np.save('../../../preds/fullconv_3/img{0}_fullconv_3'.format(FLAGS.imgn), final_mask)

  return final_mask

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  predict()

if __name__ == '__main__':
  tf.app.run()
