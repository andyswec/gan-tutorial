import logging
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np

G_INPUT_SIZE = 1
D_INPUT_SIZE = 1
D_OUTPUT_SIZE = 1
HIDDEN_SIZE = 11

BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 10 ** 6

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('gan-tutorial.log', 'w'))


def z_sample(shape):
    return np.random.uniform(0, 1, shape)


def x_sample(shape):
    return np.random.normal(-1, 1, shape)


def generator(z):
    hidden = fully_connected(z, HIDDEN_SIZE, activation_fn=tf.sigmoid)
    output = fully_connected(hidden, D_INPUT_SIZE, activation_fn=tf.sigmoid)
    return output


def discriminator(x):
    hidden = fully_connected(x, HIDDEN_SIZE, activation_fn=tf.sigmoid)
    output = fully_connected(hidden, D_OUTPUT_SIZE, activation_fn=tf.sigmoid)
    return output


with tf.device('/gpu:0'):
    with tf.variable_scope('G'):
        z = tf.placeholder(tf.float32, shape=(None, G_INPUT_SIZE))
        G = generator(z)

    with tf.variable_scope('D') as scope:
        x = tf.placeholder(tf.float32, shape=(None, D_INPUT_SIZE))
        D1 = discriminator(x)
        scope.reuse_variables()
        D2 = discriminator(G)

        loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
        loss_g = tf.reduce_mean(-tf.log(D2))

        trainable_vars = tf.trainable_variables()
        d_params = [v for v in trainable_vars if v.name.startswith('D/')]
        g_params = [v for v in trainable_vars if v.name.startswith('G/')]
        logger.info("D params count: {}".format(len(d_params)))
        logger.info("G params count: {}".format(len(g_params)))

        opt_d = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_d)
        opt_g = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_g)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
    session.run(tf.global_variables_initializer())

    for step in range(EPOCHS):
        zs = z_sample([BATCH_SIZE, G_INPUT_SIZE])
        loss_gs, _ = session.run([loss_g, opt_g], feed_dict={z: zs})

        xs = x_sample([BATCH_SIZE, D_INPUT_SIZE])
        zs = z_sample([BATCH_SIZE, G_INPUT_SIZE])
        loss_ds, _ = session.run([loss_d, opt_d], feed_dict={x: xs, z: zs})

        if step % 10000 == 0:
            logger.info('Epoch {}: loss_d={} loss_g={}'.format(step, loss_ds, loss_gs))
