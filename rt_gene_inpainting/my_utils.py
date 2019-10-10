from __future__ import print_function, division, absolute_import

import time
import numpy as np
import os
from glob import glob
from utils import *
import tensorflow as tf

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60*60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60*60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s "% self.elapsed(time.time()-self.start_time))

# cifar10 dataset -------------------------
def cifar10_process(x):
    x = x.astype(np.float32) / 127.5 - 1.
    return x

def cifar10_data():
    from keras.datasets import cifar10
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)


# mnist dataset -------------------------
def mnist_process(x):
    x = x.astype(np.float32) / 127.5 - 1.
    return x

def mnist_data():
    from keras.datasets import mnist
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


# celebA dataset -------------------------
def celebA_data(sample_idx=0):
    data = glob(os.path.join("../../../data", 'celebA_aligned', '*.png'))
    data_files = map(lambda i:data[i], sample_idx)

    data = [get_image(data_file, 
                input_height = 108,
                input_width = 108,
                resize_height = 64,
                resize_width = 64,
                is_crop=False,
                is_grayscale = False) for data_file in data_files]
    data_images = np.array(data).astype(np.float32)         

    return data_images


# PRL dataset -------------------------
def PRL_data(sample_idx=0):
    data = glob(os.path.join("../../../data/Cat/noGlasses", 'resized_64x64', '*.png'))
    data_files = map(lambda i:data[i], sample_idx)

    data = [imread_PRL(data_file,
                is_grayscale = False) for data_file in data_files]
    data_images = np.array(data).astype(np.float32)         

    return data_images

def PRL_data224(sample_idx=0):
    data = glob(os.path.join("../../../data/Cat/noGlasses", 'rgb', '*.png'))
    data_files = map(lambda i:data[i], sample_idx)

    data = [imread_PRL(data_file,
                is_grayscale = False) for data_file in data_files]
    data_images = np.array(data).astype(np.float32)         

    return data_images    

def PRL_data_image_load(data, sample_idx=0):
    data_files = map(lambda i:data[i], sample_idx)

    data = [imread_PRL(data_file,
                is_grayscale = False) for data_file in data_files]
    data_images = np.array(data).astype(np.float32)         

    return data_images  

def PRL_data224_depth(data, sample_idx=0):
    data_files = map(lambda i:data[i], sample_idx)

    data = [imread_PRL_depth(data_file) for data_file in data_files]
    data_images = np.array(data).astype(np.float32)         

    return data_images        

def PRL_data_Glasses(sample_idx=0):
    data = glob(os.path.join("../../../data/Cat", 'face_overlay', '*.png'))
    data_files = map(lambda i:data[i], sample_idx)

    data = [get_image(data_file, 
                input_height = 224,
                input_width = 224,
                resize_height = 64,
                resize_width = 64,
                is_crop=False,
                is_grayscale = False) for data_file in data_files]

    data_images = np.array(data).astype(np.float32)         

    return data_images    

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()    


def GAN_plot_images(generator, x_train, dataset='result', save2file=False, fake=True, samples=16, noise=None, step=0, folder_path = 'result'):
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channel = x_train.shape[3]
    filename = dataset+'.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = dataset+"_%05d.png" % step
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(int(np.sqrt(samples)), int(np.sqrt(samples)), i + 1)
        image = (images[i, :, :, :] + 1.) / 2.

        if channel == 1:
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        elif channel == 3:
            image = np.reshape(image, [img_rows, img_cols, channel])
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout()
    if save2file:
        plt.savefig(folder_path+'/'+filename)
        plt.close('all')
    else:
        plt.show()

def CoGAN_plot_images(generator1, generator2, x_train1, x_train2, dataset='result', save2file=False, fake=True, samples=16, noise=None, step=0, folder_path = 'result'):
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channel = x_train.shape[3]
    filename = dataset+'.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = dataset+"_%05d.png" % step
        images1 = generator1.predict(noise)
        images2 = generator2.predict(noise)
    else:
        i1 = np.random.randint(0, x_train1.shape[0], samples)
        i2 = np.random.randint(0, x_train2.shape[0], samples)
        images1 = x_train1[i1, :, :, :]
        images2 = x_train2[i2, :, :, :]

    images = np.concatenate((images1, images2), axis=0)

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(int(np.sqrt(samples)), int(np.sqrt(samples)), i + 1)
        image = (images[i, :, :, :] + 1.) / 2.

        if channel == 1:
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        elif channel == 3:
            image = np.reshape(image, [img_rows, img_cols, channel])
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout()
    if save2file:
        plt.savefig(folder_path+'/'+filename)
        plt.close('all')
    else:
        plt.show()        

def GAN_plot_images_depth(generator, x_train, dataset='result', save2file=False, fake=True, samples=16, noise=None, step=0):
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channel = 1
    filename = 'result_depth/'+dataset+'.png'
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = "result_depth/"+dataset+"_%05d.png" % step
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(int(np.sqrt(samples)), int(np.sqrt(samples)), i + 1)
        image = (images[i, :, :, :] + 1.) / 2.
        image = images[i, :, :, :]

        image = np.reshape(image, [img_rows, img_cols])
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()        


def my_save_images(images, filename, sample_num=36, save2file=True):

    img_rows = images.shape[1]
    img_cols = images.shape[2]
    channel = images.shape[3]    

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(int(np.sqrt(sample_num)), int(np.sqrt(sample_num)), i + 1)
        image = (images[i, :, :, :] + 1.) / 2.

        if channel == 1:
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        elif channel == 3:
            image = np.reshape(image, [img_rows, img_cols, channel])
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()
