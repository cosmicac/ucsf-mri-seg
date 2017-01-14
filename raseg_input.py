import os
from six.moves import xrange
import tensorflow as tf

PATCH_HEIGHT = 32
PATCH_WIDTH = 32
PATCH_DEPTH = 8
NCHANNELS = 2

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 40000

def read_train_bin(filename_queue):
  """Reads and parses patches from training data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: the spatial depth of the result (8)
      channels: number of channels
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label 0 or 1.
      int16image: a [height, width, depth, channels] int16 Tensor with the image data
  """

  class ImageRecord(object):
    pass
  result = ImageRecord()

  label_bytes = 2  # its uint16
  result.height = PATCH_HEIGHT
  result.width = PATCH_WIDTH
  result.depth = PATCH_DEPTH
  result.nchannels = NCHANNELS
  image_bytes = result.height * result.width * result.depth * result.nchannels * 2

  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # should be 32*32*8*2*2 + 2
  assert record_bytes == 32770

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of int16 that is record_bytes/2 long.
  record_bytes = tf.decode_raw(value, tf.int16)

  # The first bytes represent the label, which we convert from int16->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [1]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape.
  result.int16image = tf.reshape(tf.slice(record_bytes, [1], [int(image_bytes/2)]),
                           [result.height, result.width, result.depth, result.nchannels])

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 4-D Tensor of [height, width, depth, nchannels] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 5D tensor of [batch_size, height, width, depth, nchannels] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', tf.reshape(images[:,:,:,0,0], (batch_size, PATCH_HEIGHT, PATCH_WIDTH, 1)))

  return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
  """Construct input for network evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH, NCHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'train_and_label_2ch_batch_{0}.bin'.format(i))
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'val_and_label_2ch_batch_{0}.bin'.format(i))
                 for i in xrange(1, 2)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_train_bin(filename_queue)
  image = tf.cast(read_input.int16image, tf.float32)

  # Subtract off mean and divide by the adjusted std. of the pixels - channel wise
  mean, variance = tf.nn.moments(image, axes=[0,1,2])
  adjusted_stddev = tf.maximum(tf.sqrt(variance), tf.div(tf.constant(1.0),
                                                  tf.sqrt(tf.to_float(PATCH_HEIGHT*PATCH_WIDTH*PATCH_DEPTH))))

  # Subtract off the mean and divide by the adjusted std. of the pixels.
  # mean, variance = tf.nn.moments(image, axes=[0,1,2,3])
  # adjusted_stddev = tf.maximum(tf.sqrt(variance), tf.div(tf.constant(1.0),
  #                                                tf.sqrt(tf.to_float(tf.size(image)))))

  float_image = tf.divide(tf.subtract(image, mean), adjusted_stddev) 

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
