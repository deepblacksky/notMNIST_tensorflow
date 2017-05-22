
import tensorflow as tf


def inference(images, batch_size, n_classes):
    """
    build model of the network, compute logits 
    :param images: [batch_size, image_width, image_height, channel]
    :param batch_size: 
    :param n_classes: 
    :return: logits
    """
    print(str(images.get_shape()))
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 1, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.0, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 64, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('FC3') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        print(str(reshape.get_shape()))
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
        # L2 regularization
        weight_loss = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('lossess', weight_loss)
        biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('FC4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384, 192],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        # L2
        weight_loss = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('lossess', weight_loss)
        biases = tf.get_variable('biases', shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)

    with tf.variable_scope('FC5') as scope:
        weights = tf.get_variable('FC5',
                                  dtype=tf.float32,
                                  shape=[192, n_classes],
                                  initializer=tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc4, weights), biases, name=scope.name)

    return logits


def loss(logits, labels):
    """
    compute loss with l2 Regu
    :param logits: 
    :param labels: 
    :return: 
    """
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels,
                                                                       name='cross_entropy_pre_example')

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('lossess', cross_entropy_mean)

        total_loss = tf.add_n(tf.get_collection('lossess'), name='total_loss')

        tf.summary.scalar(scope+'/loss', total_loss)

        return total_loss


def optimize(loss, learning_rate, global_step):
    """
    use AdamOptimitizer as default
    :param loss: 
    :param learning_rate: 
    :param global_step: 
    :return: 
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def accuracy(logits, labels):
    """
    evaluate the accuracy 
    :param logits: 
    :param labels: 
    :return: 
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # because label is one-dim, so.
        # (http://stackoverflow.com/questions/37621639/tensorflow-error-minimum-tensor-rank-1-but-got-1)
        # correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels, dtype=tf.int64))
        correct = tf.cast(correct, tf.float32)
        total_accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope+'/accuracy', total_accuracy)

        return total_accuracy
