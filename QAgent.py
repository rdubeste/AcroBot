import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


class QAgent:

    def __init__(self, rate):
        # set up the neural net
        # 6 input neurons, 16 hidden layer neurons, 3 output neurons
        self.input = tf.placeholder(shape=[None, 6], dtype=tf.float64, name="input")
        self.h1 = layers.fully_connected(self.input, 16, activation_fn=tf.nn.sigmoid)
        self.h2 = layers.fully_connected(self.h1, 16, activation_fn=tf.nn.sigmoid)
        self.output = layers.fully_connected(self.h2, 3, activation_fn=tf.nn.softmax)

        # set up training procedure
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float64, name="reward_h")
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="action_h")
        # figure out which action we chose on each time step, and pass its weight into loss function
        indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), indexes)
        # loss is the mean of output times * reward
        self.loss = responsible_outputs
        # make the gradient placeholder for each variable
        trainable = tf.trainable_variables()
        count = len(trainable)
        self.gradient_placeholders = []
        for v in range(0, count):
            self.gradient_placeholders.append(tf.placeholder(tf.float64))

        # finds the gradient of our loss function with respect to each variable in trainable
        self.gradients = tf.gradients(self.loss, trainable)
        # using AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=rate)
        # using the optimizer, apply the gradient/variable pairs
        self.update = optimizer.apply_gradients(zip(self.gradient_placeholders, trainable))
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def get_output(self, state):
        return self.session.run(self.output, {self.input: [state]})

    def compute_gradients(self, states, actions, rewards):
        return self.session.run(self.gradients,
                                {self.input: states, self.action_holder: actions, self.reward_holder: rewards})

    def train(self, state):
        self.session.run(self.output, {self.input: [state]})
        print(self.session.run(tf.shape(self.output)))
