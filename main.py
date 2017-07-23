'''
InfoGAN + WGAN Model

Updated on 2017.07.22
Author : Yeonwoo Jeong
'''
from ops import mnist_for_gan, optimizer, clip, get_shape
from config import InfoGANConfig, SAVE_DIR, PICTURE_DIR
from nets import GenConv, DisConv, QConv
from utils import show_gray_image_3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
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
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)
        self.Q_rct = self.classifier(self.G_sample)
        
        self.Q_rct_classify, self.Q_rct_conti = tf.split(self.Q_rct, [10, self.c_dim-10],axis = 1)
        self.C_classify, self.C_conti = tf.split(self.C, [10, self.c_dim-10], axis = 1)
        
        self.D_loss = -tf.reduce_mean(self.D_real)+tf.reduce_mean(self.D_fake)
        self.Q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.C_classify, logits=self.Q_rct_classify))+tf.reduce_mean(tf.square(self.C_conti-self.Q_rct_conti))
        self.G_loss = -tf.reduce_mean(self.D_fake)        

        self.generator.print_vars()
        self.discriminator.print_vars()
        self.classifier.print_vars()

        self.D_optimizer = optimizer(self.D_loss, self.discriminator.vars)

        with tf.control_dependencies([self.D_optimizer]):
            self.D_optimizer_wrapped = [tf.assign(var, clip(var, -self.clip_b, self.clip_b)) for var in self.discriminator.vars]
        
        self.Q_optimizer = optimizer(self.Q_loss, self.generator.vars + self.classifier.vars)
        self.G_optimizer = optimizer(self.G_loss, self.generator.vars)

        logger.info("Building model done.")
        self.sess = tf.Session()
        
    def initialize(self):
        """Initialize all variables in graph"""
        self.sess.run(tf.global_variables_initializer())
        
    def restore(self):
        """Restore all variables in graph"""
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")     
    
    def sample_data(self, c_fix=False):
        """sampling for data
        Return:
            X_sample, z_sample, c_sample
        """
        X_sample = self.dataset(self.batch_size)
        z_sample = sample_z(self.batch_size, self.z_dim)

        if c_fix:
            conti_unit = np.linspace(-1, 1, 10)
            conti = np.transpose(np.tile(conti_unit, [2,10]))
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
                g_iter = 1
            else:
                # dynamic control
                X_sample, z_sample, c_sample = self.sample_data() 
                D_loss = self.sess.run(self.D_loss, feed_dict = {self.X : X_sample, self.Z : z_sample, self.C : c_sample})

                if abs(D_loss) < 0.01 :
                    d_iter = 25
                    g_iter = 1
                elif abs(D_loss) < 0.1:
                    d_iter = 5
                    g_iter = 1
                elif abs(D_loss) < 0.9:
                    d_iter = 5
                    g_iter = 1
                elif abs(D_loss) < 0.99:
                    d_iter = 25
                    g_iter = 1

            for _ in range(d_iter):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.D_optimizer_wrapped, feed_dict = {self.X : X_sample, self.Z : z_sample, self.C : c_sample})
            
            for _ in range(g_iter):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.G_optimizer, feed_dict = {self.Z : z_sample, self.C : c_sample})
            
            for _ in range(g_iter):
                X_sample, z_sample, c_sample = self.sample_data()
                self.sess.run(self.Q_optimizer, feed_dict = {self.Z : z_sample, self.C : c_sample})
                
            if epoch % self.log_every == self.log_every-1:
                X_sample, z_sample, c_sample = self.sample_data(c_fix = True)

                D_loss = self.sess.run(self.D_loss, feed_dict = {self.X : X_sample, self.Z : z_sample, self.C : c_sample})
                G_loss = self.sess.run(self.G_loss, feed_dict = {self.Z : z_sample, self.C : c_sample})
                Q_loss = self.sess.run(self.Q_loss, feed_dict = {self.Z : z_sample, self.C : c_sample})
                
                gray_3d = self.sess.run(self.G_sample, feed_dict = {self.Z : z_sample, self.C : c_sample}) # self.batch_size x 28 x 28 x 1
                gray_3d = np.squeeze(gray_3d)#self.batch_size x 28 x 28

                # Store generated image on PICTURE_DIR
                count+=1
                fig = show_gray_image_3d(gray_3d, col=10, figsize = (50, 50), dataformat = 'CHW')
                fig.savefig(PICTURE_DIR+"{}.png".format(str(count).zfill(3)))
                plt.close(fig)

                logger.info("Epoch({}/{}) D_loss : {}, G_loss : {}, Q_loss : {}".format(epoch+1, train_epochs, D_loss, G_loss, Q_loss))
                saver=tf.train.Saver(max_to_keep = 10)
                saver.save(self.sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)
                logger.info("Model save in %s"%SAVE_DIR)

if __name__=='__main__':
    infogan = InfoGAN()
    infogan.initialize()
    infogan.train(20000)
