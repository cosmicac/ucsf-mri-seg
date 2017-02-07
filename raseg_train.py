from datetime import datetime
import time
import tensorflow as tf

import raseg_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../models/raseg_train_kmeans_partial',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('train_from_checkpoint', False, """Whether to train from latest checkpoint""")


def train():
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels
    images, labels = raseg_model.inputs(False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = raseg_model.inference(images)

    # Calculate loss.
    loss = raseg_model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = raseg_model.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if FLAGS.train_from_checkpoint and ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(mon_sess, ckpt.model_checkpoint_path)
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
