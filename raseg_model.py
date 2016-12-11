import gzip
import os
import sys
import tensorflow as tf
import raseg_input

# flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train model using 16-bit floating point.""")
tf.app.flags.DEFINE_string('data_dir', '../data/datasets/bins', """Directory to the data binaries""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of voxel regions in our batch.""")

# constants
NUM_CLASSES = raseg_input.NUM_CLASSES
NUM_EXAMPLES_EPOCH_TRAIN = raseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_EPOCH_EVAL = raseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 12
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.05


def variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def variable_on_gpu(name, shape, initializer):
	with tf.device('/gpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var	

def variable_with_weight_decay(name, shape, stddev, wd):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def loss(logits, labels):

	# labels must be ints
	labels = tf.cast(labels, tf.int64)

	# one-hot the labels and then softmax and calculate cross entropy for each sample
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_sample')

	# average cross entropy for the batch
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	#add l2 loss and cross entropy loss for total loss
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def add_loss_summaries(total_loss):
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	for l in losses + [total_loss]:
		tf.scalar_summary(l.op.name + ' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

	return loss_averages_op

def activation_summary(x):
	tf.histogram_summary(x.op.name + '/activations', x)
	tf.scalar_summary(tensor_name + 'sparsity', tf.nn.zero_fraction(x))

def inputs(eval_data):
  """Construct input for evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 5D tensor of [batch_size, PATCH_HEIGHT, PATCH_DEPTH, NCHANNELS, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = raseg_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(voxel_regions):

	# Conv1
	with tf.variable_scope('conv1') as scope:
		kernel = variable_with_weight_decay('weights', shape=[5, 5, 5, 1, 32], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(voxel_regions, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(sums, name=scope.name)

	# Conv2
	with tf.variable_scope('conv2') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,32,32], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(conv1, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(sums, name=scope.name)

	# Convd1 (1st downsampling layer)
	with tf.variable_scope('convd1') as scope:
		kernel = variable_with_weight_decay('weights', shape=[2,2,2,32,32], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(conv2, kernel, strides=[1,2,2,2,1], padding='SAME')
		biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		convd1= tf.nn.relu(sums, name=scope.name)

	# Conv3
	with tf.variable_scope('conv3') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,32,64], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(convd1, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(sums, name=scope.name)

	# Conv4
	with tf.variable_scope('conv4') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,64,64], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(conv3, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(sums, name=scope.name)

	# Convd2 (2nd downsampling layer)
	with tf.variable_scope('convd2') as scope:
		kernel = variable_with_weight_decay('weights', shape=[2,2,2,64,64], stddev=5e-2, wd=0.00)
		conv = tf.nn.conv3d(conv4, kernel, strides=[1,2,2,2,1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		convd2 = tf.nn.relu(sums, name=scope.name)

	# Fc1 (1st fully connected layer)
	with tf.variable_scope('fc1') as scope:
		# collapse our volumes into one dimension
		collapsed = tf.reshape(convd2, [FLAGS.batch_size, -1])
		flat_dim = collapsed.get_shape()[1].value
		weights = variable_with_weight_decay('weights', shape=[flat_dim, 4096], stddev=0.04, wd=0.004)
		biases = variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(collapsed, weights) + biases, name=scope.name)

	# Fc2 (2nd fully connected layer)
	with tf.variable_scope('fc2') as scope:
		weights = variable_with_weight_decay('weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
		biases = variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
		fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

	# output, sums before softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = variable_with_weight_decay('weights', [4096, NUM_CLASSES], stddev=1/4096.0, wd=0.00)
		biases = variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return softmax_linear

def train(total_loss, global_step):
	num_batches_per_epoch = NUM_EXAMPLES_EPOCH_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# decay learning rate
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

	tf.scalar_summary('learning_rate', lr)

	# moving averages
	loss_averages_op = add_loss_summaries(total_loss)

	# compute gradients
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# apply gradients
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# add histograms for trainable variables
	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	# add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

	# track the moving averages of all trainable variables
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op  = variable_averages.apply(tf.trainable_variables())

	# construct train operation
	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
