from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import raseg_model
import util.make_predsets as mp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../models/raseg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'train_eval' or 'test'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../models/raseg_train_t2bmeonly',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 262144,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('predict_slice', True,
                         """Whether to predict a slice and save predictions""")


def eval_once(saver, summary_writer, ops, summary_op, nexamples=None):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    ops: Ops to run
    summary_op: Summary op.
  """

  preds = []
  # logits = []

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      #saver.restore(sess, '../models/raseg_train_2ch/model.ckpt-17312')

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/raseg_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      if not nexamples:
        nexamples = FLAGS.num_examples

      num_iter = int(math.ceil(nexamples / FLAGS.batch_size))
      true_count = 0  # Count sthe number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      while step < num_iter and not coord.should_stop():
        # Run the ops and whatever is needed to return their output
        predictions = sess.run(ops)
        # Extend the predictions array with the predictions from the current batch
        if FLAGS.predict_slice:
          preds.extend(predictions[1].indices.flatten())
        # preds.append(predictions[1].indices)

        # Append the logits for the current batch
        # logits.append(predictions[1].values)

        # Increase the count of correct predictions
        true_count += np.sum(predictions[0])
        #true_count += np.sum(predictions)
        step += 1
        if step % 200 == 0:
          print("Processing batch {0}.".format(step))

      # Save predictions
      # np.save('../preds/img8d9_kmeans_preds', preds)

      # Save logits
      # np.save('../preds/img8d9_2ch_big_logits', logits)
      
      # Compute precision @ 1.
      print(true_count)
      print(total_sample_count)
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
      
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return preds


def evaluate():

  if FLAGS.predict_slice:
    # load images, both channels
    images_and_labels = np.load('../data/datasets/t2imgs_and_prereg_labels.npy')
    #pre_images = np.load('../data/datasets/pre_images_aligned_regfix.npy')

    # make predset and save the binary
    cluster_labels, hi_cluster = mp.predict_slice_kmeans_t2only(images_and_labels, FLAGS.imgn, FLAGS.depthn,
                                  '../data/datasets/bins/img{0}d{1}_and_label_regfix_batch_1.bin'.format(FLAGS.imgn, FLAGS.depthn))

  """Eval model for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels
    eval_data = FLAGS.eval_data == 'test'

    print(FLAGS.imgn)
    print(FLAGS.depthn)
    images, labels = raseg_model.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = raseg_model.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op_vals = tf.nn.top_k(logits, k=1)
    # top_k_op_vals = tf.nn.top_k(logits, k=2, sorted=False)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        raseg_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:

      if FLAGS.predict_slice:

        # find indexes of where we're going to predict
        i, j = np.where(cluster_labels == hi_cluster)
        preds = eval_once(saver, summary_writer, [top_k_op, top_k_op_vals], summary_op, nexamples=len(i))

        print(len(i))
        print(len(preds))
        print(np.sum(preds))

        # make the mask
        mask = np.zeros(cluster_labels.shape)
        for k in range(len(i)):
          mask[i[k], j[k]] = preds[k]

        np.save('../preds/img{0}d{1}_t2bmeonly_preds'.format(FLAGS.imgn, FLAGS.depthn), mask)

      else:
        eval_once(saver, summary_writer, [top_k_op, top_k_op_vals], summary_op)
        #eval_once(saver, summary_writer, [top_k_op], summary_op)

      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)



def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
