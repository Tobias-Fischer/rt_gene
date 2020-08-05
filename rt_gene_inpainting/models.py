from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


# LSGAN Model
class LSGAN_Model(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1, noise_dim=100, dataset='MNIST'):

        self.dataset = dataset
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.noise_dim = noise_dim

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None

        self.optimizer = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    def discriminator(self):
        if self.D:
            return self.D

        # kern_init = initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        kern_init = initializers.glorot_normal()

        input_shape = (self.img_rows, self.img_cols, self.channel)
        input_img = Input(shape=input_shape, name='Input_Image')

        x = Conv2D(16, 5, strides=2, input_shape=input_shape, padding='same', kernel_initializer=kern_init)(input_img)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(32, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(64, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(256, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Out: 1-dim probability
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        self.D = Model(inputs=input_img, outputs=x, name='Discriminator')

        self.D.summary()
        return self.D

    def generator(self):

        if self.G:
            return self.G

        kern_init = initializers.glorot_normal()

        input_shape = (self.noise_dim,)
        input_noise = Input(shape=input_shape, name='noise')

        dim = 7
        depth = 512

        x = Dense(dim * dim * depth, kernel_initializer=kern_init)(input_noise)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        x = Reshape((dim, dim, depth))(x)

        x = Conv2DTranspose(depth / 2, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = Activation('selu')(x)

        x = Conv2DTranspose(depth / 4, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = Activation('selu')(x)

        x = Conv2DTranspose(depth / 8, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = Activation('selu')(x)

        x = Conv2DTranspose(depth / 16, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        # x = BatchNormalization()(x)
        x = Activation('selu')(x)

        x = Conv2DTranspose(self.channel, 5, strides=2, padding='same', kernel_initializer=kern_init)(x)
        x = Activation('tanh')(x)                               

        self.G = Model(inputs=input_noise, outputs=x, name='Generator')

        self.G.summary()
        return self.G    

    def adversarial_model(self, gen, dis):
        if self.AM:
            return self.AM

        input_shape = (self.noise_dim,)
        input_noise_AM = Input(shape=input_shape, name='noise')         
        img_fake = gen(input_noise_AM)
        x = dis(img_fake)
        out_AM = Dropout(1.0, name='out_img_fake')(x)
        self.AM = Model(inputs=input_noise_AM, outputs=out_AM)           

        set_trainability(dis, False)

        self.AM.compile(loss=self.loss_LSGAN, optimizer=self.optimizer, metrics=['accuracy'])
        self.AM.summary()
        return self.AM

    def discriminator_model(self, dis):
        if self.DM:
            return self.DM

        input_shape = (self.img_rows, self.img_cols, self.channel)
        input_img = Input(shape=input_shape)
        x = dis(input_img)

        self.DM = Model(inputs=input_img, outputs=x)

        self.DM.compile(loss=self.loss_LSGAN, optimizer=self.optimizer, metrics=['accuracy'])
        return self.DM

    @staticmethod
    def loss_LSGAN(y_true, y_pred):
        return K.mean(K.square(y_pred-y_true), axis=-1)/2


# Completion Model
class Completion_Model(object):
    def __init__(self, noise_dim=100):
        self.noise_dim = noise_dim        

        self.CL = None

    # complete loss = contextural loss + perceptual loss
    def cal_complete_loss(self, gen, dis):
        if self.CL:
            return self.CL

        input_shape = (self.noise_dim,)
        input_noise_CL = Input(shape=input_shape, name='noise')         
        out_gen_img = gen(input_noise_CL)
        out_gen_img = Dropout(1.0, name='name_out_gen_img')(out_gen_img)        
        out_dis_val = dis(out_gen_img)

        out_dis_val = Dropout(1.0, name='name_out_dis_val')(out_dis_val)
        self.CL = Model(inputs=input_noise_CL, outputs=[out_dis_val, out_gen_img])          
        return self.CL            
