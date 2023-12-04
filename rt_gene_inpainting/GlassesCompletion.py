from __future__ import print_function, division, absolute_import

from models import LSGAN_Model, Completion_Model
from utils import *
import os
from glob import glob
from tqdm import tqdm

from tensorflow.keras import backend as K
import external.poissonblending as blending


class GlassesCompletion(object):
    def __init__(self, dataset_common_folder_path, dataset):
        epoch_num_total = {
            "s000": 100,
            "s001": 100, 
            "s002": 86,
            "s003": 100,
            "s004": 76,
            "s005": 77,
            "s006": 100,
            "s007": 81,
            "s008": 73,
            "s009": 58,
            "s010": 70, 
            "s011": 53,
            "s012": 99,
            "s013": 29,
            "s014": 74,
            "s015": 98,
            "s016": 95 
        }

        epoch_num = epoch_num_total[dataset]

        self.folder_path_images = dataset_common_folder_path + dataset + "_glasses"
        self.path_images = self.folder_path_images + "/original/face_before_inpainting"
        self.path_completion = self.folder_path_images + "/original/inpainting"

        self.path_GAN_model = dataset_common_folder_path + dataset + "_noglasses/natural/GAN/model"

        if not os.path.exists(self.path_completion):
            os.mkdir(self.path_completion)         

        if not os.path.exists(self.path_completion+'/hats'):
            os.mkdir(self.path_completion+'/hats')         

        if not os.path.exists(self.path_completion+'/blended'):
            os.mkdir(self.path_completion+'/blended')                                             

        self.img_rows = 224
        self.img_cols = 224
        self.channel = 3
        self.image_shape = [self.img_rows, self.img_cols, self.channel]

        self.dataset = dataset

        # For GAN
        self.noise_dim = 100   

        self.generator = tf.keras.models.load_model(self.path_GAN_model+"/GAN_"+str(epoch_num)+"_"+self.dataset+"_forganECCV_generator_uniform.h5")
        self.adversarial_cost = tf.keras.models.load_model(self.path_GAN_model+"/GAN_"+str(epoch_num)+"_"+self.dataset+"_forganECCV_adversarial_model_uniform.h5", custom_objects={'loss_LSGAN': loss_LSGAN})

        print('Done Loading Pre-trained Network!')

    def image_completion_random_search(self, nIter=1000, GPU_ID="0"):
        filename_total_face = sorted(glob(os.path.join(self.path_images, 'face_*.png')))

        num_total_data = len(filename_total_face)

        print(num_total_data)

        print('=======================================================')

        GAN_4_loss = LSGAN_Model(self.img_rows, self.img_cols, self.channel, self.noise_dim, self.dataset)

        dis_4_loss = GAN_4_loss.discriminator()
        gen_4_loss = GAN_4_loss.generator()

        GAN_Completion_model = Completion_Model(self.noise_dim)

        complete_loss_model = GAN_Completion_model.cal_complete_loss(gen_4_loss, dis_4_loss)

        mask_tensor = tf.compat.v1.placeholder(tf.float32, self.image_shape, name='mask')
        images_tensor = tf.compat.v1.placeholder(tf.float32, self.image_shape, name='real_images')       
        G_images_tensor = tf.compat.v1.placeholder(tf.float32, self.image_shape, name='fake_images')

        loss_contextual_temp = tf.abs(tf.multiply(mask_tensor, G_images_tensor) - tf.multiply(mask_tensor, images_tensor))

        loss_contextual = tf.reduce_sum(input_tensor=tf.reshape(loss_contextual_temp, [tf.shape(input=loss_contextual_temp)[0], -1]), axis=1)
        loss_perceptual = complete_loss_model.output[0]
        loss = loss_contextual + 0.1*loss_perceptual

        gradients = K.gradients(loss, complete_loss_model.input)
        print('gradients: ', gradients)

        print('=======================================================')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.visible_device_list = GPU_ID
        config.gpu_options.allow_growth = True

        sess = tf.compat.v1.Session(config=config)    
        sess.run(tf.compat.v1.global_variables_initializer())

        print(self.path_completion)

        for img_idx in tqdm(range(0, num_total_data)):
            filename_face = filename_total_face[img_idx]
            filename_index = filename_face[-14:-8]
            filename_mask = self.folder_path_images + '/original/mask/mask_' + filename_index + '_overlay.png'   

            filename_out = self.path_completion+'/blended/' + filename_index + '.png'
            if os.path.isfile(filename_out):
                continue

            data_face = imread_PRL(filename_face, is_grayscale=False)
            image_face = np.array(data_face).astype(float)   

            data_mask = imread_PRL(filename_mask, is_grayscale=True)
            image_mask = np.array(data_mask).astype(float)                            

            # Sample index
            sample_num = 1
            # sample_noise_input = np.random.uniform(-1.0, 1.0, size=[sample_num, self.noise_dim])

            # mask generation
            mask = self.mask_PRL_Glasses(image_mask)

            # masked_images = np.multiply(image_face, mask)

            # y = np.ones([sample_num, 1])
            zhats = np.random.uniform(-1.0, 1.0, size=[sample_num, self.noise_dim])

            # loss_buf = 0

            l_buf = 10000000
            zhats_buf = zhats
            # final_iter = 0

            for j in range(nIter):
                zhats_search = np.random.uniform(-1.0, 1.0, size=[sample_num, self.noise_dim])
                G_imgs = self.generator.predict(zhats_search)
                G_imgs = np.squeeze(G_imgs)
                g, l, lc, lp = sess.run([gradients, loss, loss_contextual, loss_perceptual], feed_dict={complete_loss_model.input: zhats_search, mask_tensor: mask, images_tensor: image_face, G_images_tensor: G_imgs})

                if np.sum(l) < l_buf:
                    l_buf = np.sum(l)
                    zhats_buf = zhats_search
                    # final_iter = j

            zhats = zhats_buf
            G_imgs = self.generator.predict(zhats)
            G_imgs = np.squeeze(G_imgs)

            # --------------------------------------------------------------
            # Generate completed images 
            # inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
            # completed = masked_images + inv_masked_hat_images

            filename = self.path_completion+'/hats/' + filename_index + '.png'
            scipy.misc.imsave(filename, (G_imgs + 1) / 2)

            # Poisson Blending
            image_out = self.iminvtransform(G_imgs)
            image_in = self.iminvtransform(image_face)

            try:          
                image_out = self.poissonblending(image_in, image_out, mask)
                filename = self.path_completion+'/blended/' + filename_index + '.png'
                scipy.misc.imsave(filename, image_out)
            except:
                print("Error occurred while blending: " + str(filename_index))
                pass       

        sess.close()

    def mask_PRL_Glasses(self, mask_images):
        mask = np.ones(self.image_shape)

        for ir in range(self.img_rows):
            for ic in range(self.img_cols):
                # if mask_images[ir,ic] >= (127.5/127.5-1):
                if mask_images[ir, ic] > (0/127.5-1):
                    mask[ir, ic, :] = 0
                    
        return mask

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blending.blend(img1, img2, 1 - mask)

    @staticmethod
    def iminvtransform(img):
        """Helper: Rescale pixel value ranges to 0 and 1"""
        return (np.array(img) + 1.0) / 2.0


def loss_LSGAN(y_true, y_pred):
    return K.mean(K.square(y_pred-y_true), axis=-1)/2
