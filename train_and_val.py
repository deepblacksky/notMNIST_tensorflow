# DATA:
# 1. notMNIST(TFRecords binary version):http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html

# TO Train and test:
# 0. get data ready
# 1. run training_and_val.py and call train() in the console
# 2. call evaluate() in the console to test

import os
import os.path
import math

import numpy as np
import tensorflow as tf

import model
import notmnist


IMG_W = 28
IMG_H = 28
N_CLASSES = 10
BATCH_SIZE = 128
learning_rate = 0.001
MAX_STEP = 15000

DATA_DIR = '/home/yuxin/data/notMNIST/notMNIST.pickle'


def train():

    train_log_dir = './log/train/'
    val_log_dir = './log/val'

    with tf.name_scope('input'):
        notmnist_data = notmnist.load_data(DATA_DIR, one_hot=True)

    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_W * IMG_H])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, N_CLASSES])

    logits = model.inference(x_image, BATCH_SIZE, N_CLASSES)
    loss = model.loss(logits, y_)
    accuracy = model.accuracy(logits, y_)
    my_global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = model.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):

            # if coord.should_stop():
            #     break

            train_images, train_labels = notmnist_data.train.next_batch(BATCH_SIZE)
            # print(train_images.shape, ", ", type(train_images))
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={x: train_images, y_: train_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={x: train_images, y_: train_labels})
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = notmnist_data.validation.next_batch(BATCH_SIZE)
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print("** Step: %d, loss: %.2f, accuracy: %.2f%%" % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training!')
    finally:
        # coord.request_stop()
        pass

    # coord.join(threads)
    sess.close()


def evaluate():

    with tf.Graph().as_default():

        log_dir = './log/train/'
        n_test = 10000

        notmnist_test = notmnist.load_data(DATA_DIR, one_hot=True)

        x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_W * IMG_H])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, N_CLASSES])
        # y_label = tf.cast(y_, tf.int32)

        logits = model.inference(x_image, BATCH_SIZE, N_CLASSES)
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y_, 1))
        correct = tf.cast(correct, tf.int32)
        n_correct = tf.reduce_sum(correct)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading success, global step: %s' % global_step)
            else:
                print('Not found checkpoint file')
                return

            try:
                print('Evaluating...')
                num_step = int(math.ceil(n_test / BATCH_SIZE))
                num_example = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step:
                    test_images, test_labels = notmnist_test.test.next_batch(BATCH_SIZE)
                    batch_correct = sess.run([n_correct],
                                             feed_dict={x: test_images, y_: test_labels})
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' % num_example)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100 * total_correct / num_example))
            except Exception as e:
                print(e)
            finally:
                pass


if __name__ == '__main__':
    # train()
    evaluate()



