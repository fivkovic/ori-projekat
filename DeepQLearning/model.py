import numpy
import tensorflow
import random
from functools import reduce

def max_pool_2x2(x, kernel_shape, name):
    ksize = [1, *kernel_shape, 1]
    strides = [1, *kernel_shape, 1]
    return tensorflow.nn.max_pool(x, ksize, strides, padding='SAME', name=name)


def conv2d(x, output_dim, kernel_shape, stride, name):
    stride = [1, stride[0], stride[1], 1]

    with tensorflow.variable_scope(name):
        w = tensorflow.Variable(tensorflow.truncated_normal(kernel_shape, mean=0, stddev=.1), dtype=tensorflow.float32, name="w")
        conv = tensorflow.nn.conv2d(x, w, stride, "VALID")
        b = tensorflow.Variable(tensorflow.constant(0.1, shape=[output_dim]), name="b")
        out = tensorflow.nn.bias_add(conv, b)
        out = tensorflow.nn.relu(out)

    return out, w, b


def linear(x, output_size, name, activation_fn=tensorflow.nn.relu):
    shape = x.get_shape().as_list()

    with tensorflow.variable_scope(name):
        w = tensorflow.Variable(tensorflow.random_normal([shape[1], output_size], stddev=.02), dtype=tensorflow.float32, name='w')
        b = tensorflow.Variable(tensorflow.zeros([output_size]), name='b')
        out = tensorflow.nn.bias_add(tensorflow.matmul(x, w), b)

        if activation_fn != None:
            out =  activation_fn(out)

    return out, w, b


class DeepQNetwork:

    def __init__(self, session, height, width, num_actions, name, writer=None):
        self.num_actions = num_actions
        self.height = height
        self.width = width
        self.name = name
        self.vars = []
        self.session = session

        self.summary_ops = []
        self.__initialize()
        self.writer = writer


    def get_action_and_q(self, states):
        states = states.reshape(-1, 4, self.height, self.width)
        return self.session.run([self.a, self.Q], {self.state: states})

    def get_action(self, states):
        states = states.reshape(-1, 4, self.height, self.width)
        num_states = states.shape[0]
        actions = self.session.run(self.a, {self.state: states})
        return actions[0] if num_states == 1 else actions

    def perform_training(self, states, actions, targets, cnt):
        states = states.reshape(-1, 4, self.height, self.width)
        feed_dict = {self.state: states, self.actions: actions, self.Q_target: targets}
        summary,_ = self.session.run([tensorflow.summary.merge(self.summary_ops), self.minimize], feed_dict)
        if self.writer: self.writer.add_summary(summary, global_step=cnt)

    def copy_variables(self, other):
        ops = []
        for var_self, var_other in zip(self.vars, other.vars):
            ops.append(var_self.assign(var_other.value()))

        self.session.run(ops)

    def __initialize(self):

        with tensorflow.variable_scope(self.name):
            # batchsize x memory x height x width
            self.state =  tensorflow.placeholder(shape=[None, 4, self.height, self.width], dtype=tensorflow.float32)
            # batchsize x height x width x memory
            self.state_perm = tensorflow.transpose(self.state, perm=[0, 2, 3, 1])
            self.summary_ops.append(tensorflow.summary.image("states", self.state[:, 0, :, :][..., tensorflow.newaxis], max_outputs=10))

            conv1, w1, b1 = conv2d(self.state_perm, 32, [8, 8, 4, 32], [4, 4], "conv1")
            max_pool = max_pool_2x2(conv1, [2, 2], "maxpool")
            conv2, w2, b2 = conv2d(max_pool, 64, [4, 4, 32, 64], [2, 2], "conv2")
            conv3, w3, b3 = conv2d(conv2, 64, [3, 3, 64, 64], [1, 1], "conv3")
            self.vars += [w1, b1, w2, b2, w3, b3]

            shape = conv3.get_shape().as_list()
            conv3_flat = tensorflow.reshape(conv3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            # Dueling
            value_hid, w4, b4 = linear(conv3_flat, 512, "value_hid")
            adv_hid, w5, b5 = linear(conv3_flat, 512, "adv_hid")

            value, w6, b6 = linear(value_hid, 1, "value", activation_fn=None)
            advantage, w7, b7 = linear(adv_hid, self.num_actions, "advantage", activation_fn=None)
            self.vars += [w4, b4, w5, b5, w6, b6, w7, b7]

            # Average Dueling
            self.Qs = value + (advantage - tensorflow.reduce_mean(advantage, axis=1, keepdims=True))

            # action with highest Q values
            self.a = tensorflow.argmax(self.Qs, 1)
            # Q value belonging to selected action
            self.Q = tensorflow.reduce_max(self.Qs, 1)
            tensorflow.summary.scalar("Q", self.Q)

            # For training
            self.Q_target = tensorflow.placeholder(shape=[None], dtype=tensorflow.float32)
            self.actions = tensorflow.placeholder(shape=[None], dtype=tensorflow.int32)
            actions_onehot = tensorflow.one_hot(self.actions, self.num_actions, on_value=1., off_value=0., axis=1, dtype=tensorflow.float32)

            Q_tmp = tensorflow.reduce_sum(tensorflow.multiply(self.Qs, actions_onehot), axis=1)
            loss = tensorflow.reduce_mean(tensorflow.square(self.Q_target - Q_tmp))
            self.summary_ops.append(tensorflow.summary.scalar("loss", loss))
            optimizer = tensorflow.train.AdamOptimizer()
            self.minimize = optimizer.minimize(loss)
