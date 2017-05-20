from __future__ import division
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import raseg_model
import util.make_predsets as mp

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../../../models/raseg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'train_eval' or 'test'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../../../models/raseg_train_fullimg_3',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 8,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 8,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

def eval_once(saver, summary_writer, ops, summary_op, nexamples=None):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    ops: Ops to run
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)

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
      dsc = []  # Store the DSC of each val image.
      step = 0

      while step < num_iter and not coord.should_stop():
        # Run the ops and whatever is needed to return their output
        dice_coeff = sess.run(ops)
        print('Validation image {0}, DSC: {1}'.format(step, dice_coeff))
        dsc.append(dice_coeff)
        step += 1

      # Compute precision @ 1.
      mean_dsc = np.asscalar(np.mean(dsc))
      print('%s: Mean DSC = %.4f' % (datetime.now(), mean_dsc))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Mean DSC', simple_value=mean_dsc)
      summary_writer.add_summary(summary, global_step)
      
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return mean_dsc

def evaluate():

  """Eval model for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels
    eval_data = FLAGS.eval_data == 'test'

    images, labels = raseg_model.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = raseg_model.inference(images)
    dice_coeff = raseg_model.dice_coeff(logits, labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        raseg_model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, [dice_coeff], summary_op)
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
