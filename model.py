import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train model using 16-bit floating point.""")
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of voxel regions in our batch.""")

# constants
NUM_CLASSES = 2

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
	var = variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def forward_pass(voxel_regions):

	# Conv1
	with tf.variable_scope('conv1') as scope:
		kernel = variable_with_weight_decay('weights', shape=[5, 5, 5, 1, 32], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(voxel_regions, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
		biases = variable_on_gpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(sums, name=scope.name)

	# Conv2
	with tf.variable_scope('conv2') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,32,32]), stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(conv1, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_gpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(sums, name=scope.name)

	# Convd1 (1st downsampling layer)
	with tf.variable_scope('convd1') as scope:
		kernel = variable_with_weight_decay('weights', shape=[2,2 2,32,32]), stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(conv2, kernel, strides=[1,2,2,2,1], padding='SAME')
		biases = variable_on_gpu('biases', [32], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		convd1= tf.nn.relu(sums, name=scope.name)

	# Conv3
	with tf.variable_scope('conv3') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,32,64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(convd1, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_gpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(sums, name=scope.name)

	# Conv4
	with tf.variable_scope('conv4') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3,3,3,64,64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(conv3, kernel, strides=[1,1,1,1,1], padding='SAME')
		biases = variable_on_gpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(sums, name=scope.name)

	# Convd2 (2nd downsampling layer)
	with tf.variable_scope('convd2') as scope:
		kernel = variable_with_weight_decay('weights', shape=[2,2,2,64,64])
		conv = tf.nn.conv3d(conv4, kernel, strides=[1,2,2,2,1])
		biases = variable_on_gpu('biases', [64], tf.constant_initializer(0.0))
		sums = tf.nn.bias_add(conv, biases)
		convd2 = tf.nn.relu(sums, name=scope.name)

 	# Fc1 (1st fully connected layer)
 	with tf.variable_scope('fc1') as scope:
 		# collapse our volumes into one dimension
 		collapsed = tf.reshape(convd2, [FLAGS.batch_size, -1])
 		flat_dim = collapsed.get_shape()[1].value
 		weights = variable_with_weight_decay('weights', shape=[flat_dim, 4096], stddev=0.04, wd=0.004)
		biases = variable_on_gpu('biases', [4096], tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(collapsed, weights) + biases, name=scope.name)

	# Fc2 (2nd fully connected layer)
	with tf.variable_scope('fc2') as scope:
		weights = variable_with_weight_decay('weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
		biases = variable_on_gpu('biases', [4096], tf.constant_initializer(0.1))
		fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

	# output, sums before softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = variable_with_weight_decay('weights', [4096, NUM_CLASSES], stddev=1/4096.0, wd=0.0)
		biases = variable_on_gpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return softmax_linear
