from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import raseg_model
import util.make_predsets as mp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../../../models/raseg_predict',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../models/raseg_fullconv',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 262144,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

def predict():

  # load images, both channels
  images_and_labels = np.load('../../../data/datasets/images_and_labels.npy')
  pre_images = np.load('../data/datasets/pre_images_.npy')

  img = images_and_labels[FLAGS.imgn,0,:,:,:]
  pre_img = pre_images[FLAGS.imgn,:,:,:]
  labs = images_and_labels[FLAGS.imgn,1,:,:,:]

  # Indexes for the 32 patches to predict
  patches = np.zeros((32,128,128,16,2))
  depth_idx = [8,12]
  hw_idx = [64, 192, 320, 448]

  # extract patches for classification
  k = 0
  for d in depth_idx:
    for h in hw_idx:
      for w in hw_idx:
        patches[k,:,:,:,0] = extract_patch(img, (h,w,d))
        patches[k,:,:,:,1] = extract_patch(pre_img, (h,w,d))
        k += 1

  batches = [patches[:8,:,:,:,:], patches[8:16,:,:,:,:], patches[16:24,:,:,:,:], patches[24:,:,:,:,:]]  
  preds = np.zeros((32,128,128,16))

  with tf.Graph().as_default() as g:

    # Restore the moving average version of the learaned variables for prediction.
    variable_averages = tf.train.ExponentialMovingAverage(
        raseg_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)


    with tf.Session() as sess:

      # Load checkpoing state.
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found.')
        return

      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        c = 0
        for b in batches:
          
          # convert batch to tensors and feed through cnn
          images = tf.convert_to_tensor(b, dtype=tf.float32)
          logits = raseg_model.inference(images)
          softmax = tf.nn.softmax(logits, dim=-1)
          preds_op = tf.argmax(softmax, axis=4)
  
          preds_batch = sess.run(preds_op)
          preds[8*c:8*(c+1),:,:,:] = preds_batch
          c += 1

      except Exception as e:
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  return preds

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  predict()

if __name__ == '__main__':
  tf.app.run()
