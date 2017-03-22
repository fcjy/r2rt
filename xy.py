# http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function 

import collections

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "train", "train | test")
flags.DEFINE_integer("batch_size", 10, "")
flags.DEFINE_integer("step_num", 20, "")
flags.DEFINE_integer("state_size", 4, "")
flags.DEFINE_float("learning_rate", 0.1, "")

def gen_x(size):
	return np.random.choice(2, size)

def gen_y(x):
	size = len(x)
	y = np.zeros(size, dtype=np.int32)
	for i in range(size):
		p = 0.5
		if i - 3 >= 0 and x[i - 3] == 1:
			p += 0.5
		if i - 8 >= 0 and x[i - 8] == 1:
			p -= 0.25
		if np.random.rand() < p:
			y[i] = 1
	return y

def gen_batch(x, y, batch_size, step_num):
	data_len = len(x)
	assert(data_len == len(y))

	batch_len = data_len // batch_size
	epoch_num = batch_len // step_num
	x = np.reshape(x[:batch_size * batch_len], [batch_size, batch_len])
	y = np.reshape(y[:batch_size * batch_len], [batch_size, batch_len])

	for i in range(epoch_num):
		beg = i * step_num
		end = (i + 1) * step_num
		yield x[:, beg:end], y[:, beg:end]

class Model(object):
	def __init__(self, config):
		batch_size = config.batch_size
		step_num = config.step_num
		state_size = config.state_size

		self._state = np.zeros([batch_size, state_size])

	def build_graph(self, config):
		batch_size = config.batch_size
		step_num = config.step_num
		state_size = config.state_size
		learning_rate = config.learning_rate

		self._input = tf.placeholder(tf.int32, [batch_size, step_num])
		self._target = tf.placeholder(tf.int32, [batch_size, step_num])
		self._input_state = tf.placeholder(tf.float32, [batch_size, state_size])

		rnn_inputs = tf.one_hot(self._input, 2)

		cell = tf.contrib.rnn.BasicRNNCell(state_size)
		rnn_outputs, self._output_state = tf.nn.dynamic_rnn(
			cell, rnn_inputs, initial_state=self._input_state)

		with tf.variable_scope('softmax'):
			W = tf.get_variable('W', [state_size, 2])
			b = tf.get_variable('b', [2], initializer=tf.constant_initializer(0.0))
		logits = tf.reshape(
			tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
			[batch_size, step_num, 2])
		predictions = tf.nn.softmax(logits)

		self._losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self._target, logits=logits)
		self._total_losses = tf.reduce_mean(self._losses)
		self._train_step = tf.train.AdagradOptimizer(learning_rate).minimize(self._total_losses)

	def train_network(self, config):
		batch_size = config.batch_size
		step_num = config.step_num

		x = gen_x(2000000)
		y = gen_y(x)

		training_losses = []
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			tl = 0
			for step, (bx, by) in enumerate(gen_batch(x, y, batch_size, step_num)):
				_, total_losses, self._state, _ = sess.run(
					[self._losses, self._total_losses, self._output_state, self._train_step],
					feed_dict={self._input:bx, self._target:by, self._input_state:self._state})
				tl += total_losses
				if step % 100 == 0 and step > 0:
					print("Average loss at step", step, tl / 100.0)
					training_losses.append(tl / 100.0)
					tl = 0

		return training_losses

def main(_):
	m = Model(FLAGS)
	m.build_graph(FLAGS)
	training_losses = m.train_network(FLAGS)
	print(training_losses)

if __name__ == "__main__":
	tf.app.run()