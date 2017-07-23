'''
Basic operation not based on tensorflow

Updated on 2017.07.21
Author : Yeonwoo Jeong
'''

#==================================================PATH===================================================#

SAVE_DIR = './save/'
MNIST_PATH =  "../MNIST_data"
PICTURE_DIR = './asset/'

#===========================================InfoGAN configuraion===========================================#
class InfoGANConfig(object):
    def __init__(self):
        self.x_channel = 1
        self.x_size = 28
        self.x_dim = 784

        self.z_dim = 100
        self.c_dim = 12

        self.batch_size = 100
        self.log_every = 100

        self.clip_b = 0.01# clip bounday variable to be clipped to be in [-self.clip_b, self.clip_b]
