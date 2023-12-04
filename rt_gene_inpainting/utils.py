"""
Some codes from https://github.com/Newmu/dcgan_code
# Updated: 21 Feb 2017
"""
from __future__ import print_function, division, absolute_import
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def imread_PRL(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(float) / 127.5 - 1.
    else:
        return scipy.misc.imread(path).astype(float) / 127.5 - 1.


def PRL_data_image_load(data, sample_idx=0):
    data_files = map(lambda i: data[i], sample_idx)

    data = [imread_PRL(data_file, is_grayscale=False) for data_file in data_files]
    data_images = np.array(data).astype(float)

    return data_images


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def GAN_plot_images(generator, x_train, dataset='result', save2file=False, fake=True, samples=16, noise=None, step=0,
                    folder_path='result'):
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
