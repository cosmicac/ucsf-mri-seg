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
tf.app.flags.DEFINE_integer('imgn', 3, """Image number.""")
tf.app.flags.DEFINE_integer('depthn', 11, """Depth number.""")

# constants
NCHANNELS = raseg_input.NCHANNELS
NUM_CLASSES = raseg_input.NUM_CLASSES
NUM_EXAMPLES_EPOCH_TRAIN = raseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_EPOCH_EVAL = raseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 4
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
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def dice_coeff_loss(logits, labels): 

    # softmax to get probabilities
    softmax = tf.nn.softmax(logits, name='softmax')

    # argmax to get predictions from largest probability
    preds = tf.nn.argmax(softmax, axis=4, name='preds')

    # calculate intersection and both sums for every patch
    intersection = tf.reduce_sum(tf.multiply(preds, labels), axis=[1,2,3])
    preds_sum = tf.reduce_sum(preds, axis=[1,2,3])
    labels_sum = tf.reduce_sum(labels, axis=[1,2,3])

    # dice coeffs for every patch
    dice_coeff = tf.truediv(tf.multiply(intersection, 2), tf.add(preds_sum, labels_sum), name='dice_coeff_per_sample')

    # average dice coefficient for batch
    dice_coeff_mean = tf.reduce_mean(dice_coeff, name='dice_coeff')

    # loss is just negative dice coefficient
    dice_coeff_mean_loss = tf.negative(dice_coeff_mean, name='dice_coeff_loss')
    tf.add_to_collection('losses', dice_coeff_mean_loss)
    
    # add l2 loss and dice coefficient loss for total loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss(logits, labels):

	# labels must be ints
	labels = tf.cast(labels, tf.int64)

	# one-hot the labels and then softmax and calculate cross entropy for each sample
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_sample')

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
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op 
def activation_summary(x):
	tf.summary.histogram(x.op.name + '/activations', x)
	tf.summary.scalar(tensor_name + 'sparsity', tf.nn.zero_fraction(x))

def distorted_inputs():
  """Construct input for evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 5D tensor of [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH, NCHANNELS] size.
    labels: Labels. 4D tensor of [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH] size.
  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = raseg_input.distorted_inputs(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inputs(eval_data):
  """Construct input for evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 5D tensor of [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH, NCHANNELS] size.
    labels: Labels. 4D tensor of [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH] size.
  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images, labels = raseg_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size,
                                        imgn=FLAGS.imgn,
                                        depthn=FLAGS.depthn)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inference(voxel_regions):

    # height 0, convolution 1
    with tf.variable_scope('h0_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 5, NCHANNELS, 32], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(voxel_regions, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	h0_conv1 = tf.nn.relu(sums, name=scope.name)
    
    # height 0, convolution 2
    with tf.variable_scope('h0_conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 5, 32, 32], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(h0_conv1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	h0_conv2 = tf.nn.relu(sums, name=scope.name)
 
    # downsampling from height 0 to height 1
    with tf.variable_scope('down1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 32, 32], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(h0_conv2, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	down1 = tf.nn.relu(sums, name=scope.name)

    # height 1, convolution 1
    with tf.variable_scope('h1_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 32, 64], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(down1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h1_conv1 = tf.nn.relu(sums, name=scope.name)

    # height 1, convolution 2
    with tf.variable_scope('h1_conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 64], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h1_conv1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h1_conv2 = tf.nn.relu(sums, name=scope.name)

    # downsampling from height 1 to height 2
    with tf.variable_scope('down2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 64, 64], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(h1_conv2, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	down2 = tf.nn.relu(sums, name=scope.name)

    # height 2, convolution 1
    with tf.variable_scope('h2_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 128], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(down2, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h2_conv1 = tf.nn.relu(sums, name=scope.name)

    # height 2, convolution 2
    with tf.variable_scope('h2_conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 128, 128], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h2_conv1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h2_conv2 = tf.nn.relu(sums, name=scope.name)

    # downsampling from height 2 to height 3
    with tf.variable_scope('down3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 128, 128], stddev=5e-2, wd=0.00)
    	conv = tf.nn.conv3d(h2_conv2, kernel, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	down3 = tf.nn.relu(sums, name=scope.name)

    # height 3, convolution 1
    with tf.variable_scope('h3_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 128, 256], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(down3, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h3_conv1 = tf.nn.relu(sums, name=scope.name)

    # height 3, convolution 2
    with tf.variable_scope('h3_conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 256, 256], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h3_conv1, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h3_conv2 = tf.nn.relu(sums, name=scope.name)

    # upsampling from height 3 to height 2 and feed height 2 forward
    with tf.variable_scope('up1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 256, 128], stddev=5e-2, wd=0.00)
        output_shape = tf.constant([flags.BATCH_SIZE, 32, 32, 4, 128])
    	conv = tf.nn.conv3d_transpose(h3_conv2, kernel, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	up1 = tf.nn.relu(sums, name=scope.name)
        up1_concat = tf.concat([h2_conv2, up1], 4)

    # height 2, convolution 3
    with tf.variable_scope('h2_conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 256, 128], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(up1_concat, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h2_conv3 = tf.nn.relu(sums, name=scope.name)

    # height 2, convolution 4
    with tf.variable_scope('h2_conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 128, 128], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h2_conv3, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h2_conv4 = tf.nn.relu(sums, name=scope.name)

    # upsampling from height 2 to height 1 and feed height 1 forward
    with tf.variable_scope('up2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 128, 64], stddev=5e-2, wd=0.00)
        output_shape = tf.constant([flags.BATCH_SIZE, 64, 64, 8, 64])
    	conv = tf.nn.conv3d_transpose(h2_conv4, kernel, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	up2 = tf.nn.relu(sums, name=scope.name)
        up2_concat = tf.concat([h1_conv2, up2], 4)

    # height 1, convolution 3
    with tf.variable_scope('h1_conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 128, 64], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(up2_concat, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h1_conv3 = tf.nn.relu(sums, name=scope.name)

    # height 1, convolution 4
    with tf.variable_scope('h1_conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 64], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h1_conv3, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h1_conv4 = tf.nn.relu(sums, name=scope.name)

    # upsampling from height 1 to height 0 and feed height 0 forward
    with tf.variable_scope('up3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[2, 2, 2, 64, 32], stddev=5e-2, wd=0.00)
        output_shape = tf.constant([flags.BATCH_SIZE, 128, 128, 16, 32])
    	conv = tf.nn.conv3d_transpose(h1_conv4, kernel, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
    	up3 = tf.nn.relu(sums, name=scope.name)
        up3_concat = tf.concat([h0_conv2, up3], 4)

    # height 0, convolution 3
    with tf.variable_scope('h0_conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 32], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(up3_concat, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h0_conv3 = tf.nn.relu(sums, name=scope.name)

    # height 0, convolution 4
    with tf.variable_scope('h0_conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 3, 32, 32], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h0_conv3, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        h0_conv4 = tf.nn.relu(sums, name=scope.name)

    # output, logits
    with tf.variable_scope('logits') as scope:
        kernel = variable_with_weight_decay('weights', shape=[1, 1, 1, 32, 2], stddev=5e-2, wd=0.00)
        conv = tf.nn.conv3d(h0_conv4, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    	biases = variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    	sums = tf.nn.bias_add(conv, biases)
        logits = tf.nn.relu(sums, name=scope.name)
  
    return logits 

def train(total_loss, global_step):
	num_batches_per_epoch = NUM_EXAMPLES_EPOCH_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# decay learning rate
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

	tf.summary.scalar('learning_rate', lr)

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
		tf.summary.histogram(var.op.name, var)

	# add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	# track the moving averages of all trainable variables
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op  = variable_averages.apply(tf.trainable_variables())

	# construct train operation
	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
