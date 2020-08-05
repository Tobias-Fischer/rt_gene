from __future__ import print_function, division, absolute_import

from models import LSGAN_Model, set_trainability
import os
from glob import glob
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from tqdm.auto import tqdm
from utils import PRL_data_image_load, write_log, GAN_plot_images


class GAN_train(object):
    def __init__(self, dataset_common_folder_path, dataset):
        self.dataset_folder_path = dataset_common_folder_path + dataset + "_noglasses"
        self.dataset_path_images = self.dataset_folder_path + '/natural/face_before_inpainting'
        self.dataset_path_GAN = self.dataset_folder_path + '/natural/GAN'
        self.dataset_path_GAN_model = self.dataset_path_GAN + '/model'
        self.dataset_path_GAN_samples = self.dataset_path_GAN + '/samples'

        if not os.path.exists(self.dataset_path_GAN):
            os.mkdir(self.dataset_path_GAN)

        if not os.path.exists(self.dataset_path_GAN_model):
            os.mkdir(self.dataset_path_GAN_model)          

        if not os.path.exists(self.dataset_path_GAN_samples):
            os.mkdir(self.dataset_path_GAN_samples)                  

        print(self.dataset_path_images)

        # Training image setting
        self.data = glob(os.path.join(self.dataset_path_images, 'face_*.png'))
        self.num_total_data = len(self.data)
        print(self.num_total_data)

        sample_idx = np.random.randint(0, self.num_total_data, size=1)        

        self.x_train = PRL_data_image_load(self.data, sample_idx=sample_idx)
          
        print(self.x_train.shape)

        self.dataset = dataset

        self.img_rows = self.x_train.shape[1]
        self.img_cols = self.x_train.shape[2]
        self.channel = self.x_train.shape[3]

        # For GAN
        self.noise_dim = 100        

        self.GAN = LSGAN_Model(self.img_rows, self.img_cols, self.channel, self.noise_dim, dataset)

        self.discriminator = self.GAN.discriminator()
        self.generator = self.GAN.generator()
        self.discriminator_cost = self.GAN.discriminator_model(self.discriminator)        
        self.adversarial_cost = self.GAN.adversarial_model(self.generator, self.discriminator)        

        # For tensorboard callbacks
        now = datetime.now()
        log_path = "tensorboard/GAN_"+now.strftime("%Y%m%d-%H%M%S")
        self.callback = TensorBoard(log_path)
        self.callback.set_model(self.adversarial_cost)
        self.train_names = ['d_loss_real', 'd_loss_fake', 'd_loss', 'a_loss']

    def train(self, num_epoch=2000, batch_size=256, save_interval=0):
        # Initial Update Data Generation --------------------------------------
        sample_noise_input = None
        if save_interval > 0:
            # sample_noise_input = np.random.normal(0.0, 1.0, size=[36, self.noise_dim])
            sample_noise_input = np.random.uniform(-1.0, 1.0, size=[36, self.noise_dim])

        sample_idx = np.random.permutation(self.num_total_data)
        sample_idx = sample_idx[0:batch_size]

        images_train = PRL_data_image_load(self.data, sample_idx=sample_idx)            

        # noise = np.random.normal(0.0, 1.0, size=[batch_size, self.noise_dim])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_dim])
        images_fake = self.generator.predict(noise)

        # Initial Update Discriminator
        set_trainability(self.discriminator, True)
        d_loss_real = self.discriminator_cost.train_on_batch(images_train, np.ones([batch_size, 1]))
        d_loss_fake = self.discriminator_cost.train_on_batch(images_fake, np.zeros([batch_size, 1]))
        d_loss = d_loss_real + d_loss_fake

        # TRAINING STEPS ------------------------------------------------------
        print('========= Main LSGAN Training ==========')        
        num_batch = self.num_total_data // batch_size           

        for e in range(num_epoch):
            shuffled_sample_idx = np.random.permutation(self.num_total_data)

            for b in tqdm(range(num_batch)):
                batch_sample_idx = shuffled_sample_idx[b*batch_size:(b+1)*batch_size]

                images_train = PRL_data_image_load(self.data, sample_idx=batch_sample_idx)

                # noise = np.random.normal(0.0, 1.0, size=[batch_size, self.noise_dim])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_dim])
                images_fake = self.generator.predict(noise)

                # Update Discriminator
                set_trainability(self.discriminator, True)
                d_loss_real = self.discriminator_cost.train_on_batch(images_train, np.ones([batch_size, 1]))
                d_loss_fake = self.discriminator_cost.train_on_batch(images_fake, np.zeros([batch_size, 1]))
                d_loss = d_loss_real + d_loss_fake

                # Update Generator
                y = np.ones([batch_size, 1])
                # noise = np.random.normal(0.0, 1.0, size=[batch_size, self.noise_dim])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_dim])
                set_trainability(self.discriminator, False)
                a_loss = self.adversarial_cost.train_on_batch(noise, y)
                # noise = np.random.normal(0.0, 1.0, size=[batch_size, self.noise_dim])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_dim])
                set_trainability(self.discriminator, False)
                a_loss = self.adversarial_cost.train_on_batch(noise, y)

                # Log messages
                write_log(self.callback, 'D_Loss', d_loss, e*num_batch+b)
                write_log(self.callback, 'A_Loss', a_loss, e*num_batch+b)

            if save_interval > 0:
                if (e + 1) % save_interval == 0:

                    log_mesg = "Epoch %d [D loss real: %f, acc real: %f]" % (e+1, d_loss_real[0], d_loss_real[1])
                    log_mesg = "%s: [D loss fake: %f, acc fake: %f]" % (log_mesg, d_loss_fake[0], d_loss_fake[1])
                    log_mesg = "%s: [D loss: %f, acc: %f]" % (log_mesg, d_loss[0], d_loss[1])
                    log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                    print(log_mesg)      

                    GAN_plot_images(generator=self.generator, x_train=self.x_train, dataset=self.dataset,
                                    save2file=True, samples=sample_noise_input.shape[0], noise=sample_noise_input,
                                    step=(e + 1), folder_path=self.dataset_path_GAN_samples)
                    GAN_plot_images(generator=self.generator, x_train=self.x_train, dataset=self.dataset,
                                    save2file=False, samples=sample_noise_input.shape[0], noise=sample_noise_input,
                                    step=(e + 1))

            # Save trained models
            self.adversarial_cost.save(self.dataset_path_GAN_model+"/GAN_"+str(e+1)+"_"+self.dataset+"_forganECCV_adversarial_model_uniform.h5")
            self.discriminator.save(self.dataset_path_GAN_model+"/GAN_"+str(e+1)+"_"+self.dataset+"_forganECCV_discriminator_uniform.h5")
            self.generator.save(self.dataset_path_GAN_model+"/GAN_"+str(e+1)+"_"+self.dataset+"_forganECCV_generator_uniform.h5")        
