'''
InfoGAN + WGAN-gp(weight clipping) Model

Updated on 2017.07.26
Author : Yeonwoo Jeong
'''
from ops import mnist_for_gan, optimizer, clip, get_shape, softmax_cross_entropy, sigmoid_cross_entropy
from config import InfoGANConfig, SAVE_DIR, PICTURE_DIR
from utils import show_gray_image_3d, make_gif, create_dir
from nets import GenConv, DisConv, QConv
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import glob
import os


logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sample_c(c_size, add_cv, index = -1):
    '''
    Args:
        c_size - int
            number of samples
        add_v - int
            number of additional continuous random variables [-1, 1] uniform
        index - int
            default to be -1
    Return:
        [c_size, 10 + add_cv]
            10 for classification
            add_cv for independent continuos
    '''
    
    classify = np.zeros([c_size, 10])
    conti = np.random.uniform(low = -1.0, high = 1.0, size = [c_size, add_cv])
    if index < 0:
        index = np.random.randint(10)
    classify[:,index] = 1
    return np.concatenate((classify, conti), axis = 1)

def sample_z(z_size, z_dim):
    return np.random.uniform(low=-1, high=1, size= [z_size, z_dim])

class InfoGAN(InfoGANConfig):
    def __init__(self):
        InfoGANConfig.__init__(self)
        logger.info("Building model starts...")
        tf.reset_default_graph()
        self.generator = GenConv(name ='g_conv', batch_size=self.batch_size)
        self.discriminator = DisConv(name='d_conv')
        self.classifier = QConv(name='q_conv', c_dim=self.c_dim)
        self.dataset = mnist_for_gan()
        
        self.X = tf.placeholder(tf.float32, shape = [self.batch_size, self.x_size, self.x_size, self.x_channel])
        self.Z = tf.placeholder(tf.float32, shape = [self.batch_size, self.z_dim])
        self.C = tf.placeholder(tf.float32, shape = [self.batch_size, self.c_dim])
        
        self.G_sample = self.generator(tf.concat([self.Z, self.C], axis=1))
        self.epsilon = tf.random_uniform(shape=[],minval=0, maxval=1)# epsilon : sample from uniform [0,1]
        # x_hat = epsilon*x_real + (1-epsilon)*x_gen
        self.linear_ip = self.epsilon*self.X + (1-self.epsilon)*self.G_sample

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)
        self.Q_rct = self.classifier(self.G_sample)
        
        self.Q_rct_classify, self.Q_rct_conti = tf.split(self.Q_rct, [10, self.c_dim-10],axis = 1)
        self.C_classify, self.C_conti = tf.split(self.C, [10, self.c_dim-10], axis = 1)
        
        self.D_ip = self.discriminator(self.linear_ip, reuse=True)
        self.gradient = tf.gradients(self.D_ip, self.linear_ip)

        self.D_loss = -tf.reduce_mean(self.D_real)+tf.reduce_mean(self.D_fake)+self.lamb*tf.square(tf.norm(self.gradient, axis=1) - 1 )
        #self.D_loss = tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.Q_loss = tf.reduce_mean(softmax_cross_entropy(labels=self.C_classify, logits=self.Q_rct_classify))+tf.reduce_mean(tf.square(self.C_conti-self.Q_rct_conti))
        #self.G_loss = tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.G_loss = -tf.reduce_mean(self.D_fake)               

        self.generator.print_vars()
        self.discriminator.print_vars()
        self.classifier.print_vars()

        self.D_optimizer = optimizer(self.D_loss, self.discriminator.vars)
        '''
        deprecate weight clipping stead use gradient penalty stands for gp
        self.clip_b = tf.Variable(self.clip_b, trainable=False, name="clipper")
        with tf.control_dependencies([self.D_optimizer]):
            self.D_optimizer_wrapped = [tf.assign(var, clip(var, -self.clip_b, self.clip_b)) for var in self.discriminator.vars]
        '''     
        self.Q_optimizer = optimizer(self.Q_loss, self.generator.vars + self.classifier.vars)
        self.G_optimizer = optimizer(self.G_loss, self.generator.vars)

        logger.info("Building model done.")
        self.sess = tf.Session()
        
    def initialize(self):
        """Initialize all variables in graph"""
        logger.info("Initializing model parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        """Restore all variables in graph"""
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")     
    
    def sample_data(self, c_fix=False, axis=0):
        """sampling for data
        Return:
            X_sample, z_sample, c_sample
        """
        X_sample = self.dataset(self.batch_size)
        z_sample = sample_z(self.batch_size, self.z_dim)

        if c_fix:
            conti_var = np.linspace(-1, 1, 10)
            conti_fix =np.zeros(10)
            if axis == 0 :
                conti_unit = np.vstack((conti_var, conti_fix))
            else :
                conti_unit = np.vstack((conti_fix, conti_var))
            conti = np.transpose(np.tile(conti_unit, [1,10]))
            classify = list()
            for i in range(10):
                classify_unit = np.zeros(10)
                classify_unit[i] = 1
                for j in range(10):
                    classify.append(classify_unit)
            classify = np.array(classify)
            c_sample = np.concatenate((classify, conti), axis = 1)
            return X_sample, z_sample, c_sample
        else:
            c_sample = sample_c(self.batch_size, self.c_dim-10)
            return X_sample, z_sample, c_sample

    def train(self, train_epochs):
        count = 0
        for epoch in tqdm(range(train_epochs), ascii = True, desc = "batch"):
            if epoch < 25:
                d_iter = 100
            else:
                d_iter = 5
            for _ in range(d_iter):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.D_optimizer, feed_dict = {self.X : X_sample, self.Z : z_sample, self.C : c_sample})
            
            for _ in range(1):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.G_optimizer, feed_dict = {self.Z : z_sample, self.C : c_sample})
            
            for _ in range(1):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.Q_optimizer, feed_dict = {self.Z : z_sample, self.C : c_sample})
                
            if epoch % self.log_every == self.log_every-1:

                X_sample, z_sample, c_sample = self.sample_data(c_fix=True, axis = 0)
                D_loss = self.sess.run(self.D_loss, feed_dict = {self.X : X_sample, self.Z : z_sample, self.C : c_sample})
                G_loss = self.sess.run(self.G_loss, feed_dict = {self.Z : z_sample, self.C : c_sample})
                Q_loss = self.sess.run(self.Q_loss, feed_dict = {self.Z : z_sample, self.C : c_sample})
                
                count+=1
                for index in range(2):
                    X_sample, z_sample, c_sample = self.sample_data(c_fix = True, axis = index)
                    gray_3d = self.sess.run(self.G_sample, feed_dict = {self.Z : z_sample, self.C : c_sample}) # self.batch_size x 28 x 28 x 1
                    gray_3d = np.squeeze(gray_3d)#self.batch_size x 28 x 28
                	# Store generated image on PICTURE_DIR
                    fig = show_gray_image_3d(gray_3d, col=10, figsize = (50, 50), dataformat = 'CHW')
                    fig.savefig(PICTURE_DIR+"%s_%d.png"%(str(count).zfill(3), index))
                    plt.close(fig)

                logger.info("Epoch({}/{}) D_loss : {}, G_loss : {}, Q_loss : {}".format(epoch+1, train_epochs, D_loss, G_loss, Q_loss))
                
                # Save model
                saver=tf.train.Saver(max_to_keep = 10)
                saver.save(self.sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)
                logger.info("Model save in %s"%SAVE_DIR)


if __name__=='__main__':
    create_dir(SAVE_DIR)
    create_dir(PICTURE_DIR)
    infogan = InfoGAN()
    infogan.initialize()
    infogan.train(100000)

    for index in range(2):
        images_path = glob.glob(os.path.join(PICTURE_DIR, '*_%d.png'%index))
        gif_path = os.path.join(PICTURE_DIR, '%d.gif'%index)
        make_gif(sorted(images_path), gif_path)
