'''
High level networks based on basic tensorflow operations

Updated on 2017.07.21
Author : Yeonwoo Jeong
'''
from ops import fc_layer, convolution, deconvolution, get_shape, leaky_relu, flatten
import tensorflow as tf

class GenConv(object):
    def __init__(self, name = 'G_conv', batch_size = 100):
        self.name = name
        self.batch_size = batch_size

    def __call__(self, z, reuse = False):
        '''
        Args :
            z - 2D tensor [batch, zdim]
                latent vector space
            reuse - bool
                whether reuse or not
        Return :
            g - 4D tensor [batch_size, 28, 28, 1], 0 to 1
        '''
        assert get_shape(z)[0] == self.batch_size, "Batch size %d doesn't matches with %d"%(get_shape(z)[0], self.batch_size)
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            g = fc_layer(z, 7*7*128, activation = tf.nn.relu, batch_norm =False, scope = "fc1")
            g = tf.reshape(g, [-1, 7, 7, 128])
            g = deconvolution(g, [4, 4, 64, 128], output_shape = [self.batch_size, 14, 14, 64], strides = [1,2,2,1], activation = tf.nn.relu, scope = 'deconv1')
            g = deconvolution(g, [4, 4, 1, 64], output_shape = [self.batch_size, 28, 28, 1], strides = [1,2,2,1], activation = tf.nn.sigmoid, scope = 'deconv2')
            return g

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def print_vars(self):
        print(self.name)     
        print("    "+"\n    ".join(["{}:{}".format(var.name, get_shape(var)) for var in self.vars]))

class DisConv(object):
    def __init__(self, name = 'D_conv'):
        self.name = name

    def __call__(self, x, reuse=False):
        '''
        Args :
            x - 4D tensor [batch_size, 28, 28, 1]
            reuse - bool
                whether reuse or not
        Return :
            d - 2D tensor [batch, 1]
        '''
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            d = convolution(x, [4, 4, 1, 64], strides = [1,1,1,1], activation = leaky_relu, scope = 'conv1')
            d = tf.nn.max_pool(d, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
            d = convolution(d, [4, 4, 64, 128], strides = [1,1,1,1], activation = leaky_relu, scope = 'conv2')
            d = tf.nn.max_pool(d, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
            d = flatten(d)
            d = fc_layer(d, 1, scope="fc1")
        return d
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def print_vars(self):
        print(self.name)
        print("    "+"\n    ".join(["{}:{}".format(var.name, get_shape(var)) for var in self.vars]))

class QConv(object):
    def __init__(self, name = 'D_conv', c_dim = 12):
        self.name = name
        self.c_dim = c_dim

    def __call__(self, x, reuse=False):
        '''
        Args :
            x - 4D tensor [batch_size, 28, 28, 1]
            reuse - bool
                whether reuse or not
        Return :
            q - 2D tensor [batch, 12]
                10 for classification
                2 for latent code
        '''
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            q = convolution(x, [4, 4, 1, 32], strides = [1,1,1,1], activation = leaky_relu, scope = 'conv1')
            q = tf.nn.max_pool(q, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
            q = convolution(q, [4, 4, 32, 64], strides = [1,1,1,1], activation = leaky_relu, scope = 'conv2')
            q = tf.nn.max_pool(q, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
            q = flatten(q)
            q = fc_layer(q, 128, activation = leaky_relu ,scope = "d_fc1")
            q = fc_layer(q, self.c_dim, scope="d_fc2")
        return q

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def print_vars(self):
        print(self.name)
        print("    "+"\n    ".join(["{}:{}".format(var.name, get_shape(var)) for var in self.vars]))

