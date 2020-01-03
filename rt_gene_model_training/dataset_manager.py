#!/usr/bin/env python

import cv2
import numpy as np
from os import listdir

import json
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator


# size must be a tuple, (96, 96) for instance
def read_rgb_image(img_path, size, flip):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: can't read " + img_path)
    if flip:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, size, cv2.INTER_CUBIC)
#    img = img * 1./255 # normalize input
    return img


def load_one_flipped_pair(l_path, r_path, size):
    l_img = read_rgb_image(l_path, size, False)
    r_img = read_rgb_image(r_path, size, True)
    return l_img, r_img

def augment_dataset(X, Y, batch_size):
    new_X = []
    new_Y = []
    in_l = []
    in_r = []
    # create data generator
    datagen = ImageDataGenerator(
		rotation_range=10,
		zoom_range=0.1,
		width_shift_range=0.05,
		height_shift_range=0.05,
		shear_range=0.1,
		horizontal_flip=True,
		fill_mode="nearest")

    for i in range (len(Y)):
        # create iterator
        it = datagen.flow([np.array([X[0][i]]), np.array([X[1][i]])], batch_size=1)
        # generate samples
        in_r.append(X[0][i])
        in_l.append(X[1][i])
        new_Y.append(Y[i])
        for _ in range(batch_size):
            # generate batch of images
            batch = it.next()
            in_r.append(batch[0][0])
            in_l.append(batch[1][0])
            new_Y.append(Y[i])
    new_X = [np.array(in_r), np.array(in_l)]
    return new_X, new_Y

class Dataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


class CEW(Dataset):
    def __init__(self, folder_path):
        self.path = folder_path
        super(CEW, self).__init__('CEW')


class ResearcherNight(Dataset):
    def __init__(self, folder_path):
        self.path = folder_path
        super(ResearcherNight, self).__init__('RN')

    def load_pairs(self, input_size, subjects_rn15=None, subjects_rn30=None):
        if subjects_rn15 is None:
            subjects_rn15 = range(1, 10)
        if subjects_rn30 is None:
            subjects_rn30 = range(1, 18)
        in_r = []
        in_l = []
        y = []
        counts = [0, 0]
        for rn, subjects in [('rn15/', subjects_rn15), ('rn30/', subjects_rn30)]:
            for s in tqdm(subjects):
                closed_path = self.path + rn + str(s) + '/closed/'
                opened_path = self.path + rn + str(s) + '/opened/'
                closed_imgs = listdir(closed_path)
                opened_imgs = listdir(opened_path)

                for closed in closed_imgs:
                    if 'right' in closed or '.json' in closed:
                        continue
                    in_l.append(read_rgb_image(closed_path + closed, input_size, False))
                    in_r.append(read_rgb_image(closed_path + closed.replace('left', 'right'), input_size, True))
                    y.append(1.0)
                    counts[1] += 1

                for opened in opened_imgs:
                    if 'right' in opened or '.json' in opened:
                        continue
                    in_l.append(read_rgb_image(opened_path + opened, input_size, False))
                    in_r.append(read_rgb_image(opened_path + opened.replace('left', 'right'), input_size, True))
                    y.append(0.0)
                    counts[0] += 1
        x = [np.array(in_r), np.array(in_l)]
        return x, y, counts

    def load_simple_two_classes(self, input_size):
        x = []
        y = []
        counts = [0, 0]
        for rn, subjects in [('rn15/', range(1, 10)), ('rn30/', range(1, 18))]:
            for s in tqdm(subjects):
                closed_path = self.path + rn + str(s) + '/closed/'
                opened_path = self.path + rn + str(s) + '/opened/'
                closed_imgs = listdir(closed_path)
                opened_imgs = listdir(opened_path)

                for closed in closed_imgs:
                    if 'right' in closed:
                        continue
                    x.append(read_rgb_image(closed_path + closed, input_size, False))
                    x.append(read_rgb_image(closed_path + closed.replace('left', 'right'), input_size, True))
                    y.append([1.0, 0.0])
                    y.append([1.0, 0.0])
                    counts[1] += 1

                for opened in opened_imgs:
                    if 'right' in opened:
                        continue
                    x.append(read_rgb_image(opened_path + opened, input_size, False))
                    x.append(read_rgb_image(opened_path + opened.replace('left', 'right'), input_size, True))
                    y.append([0.0, 1.0])
                    y.append([0.0, 1.0])
                    counts[0] += 1

        return np.array(x), np.array(y), counts

    def load_gts(self):
        subjects_rn = {}
        for rn, subjects_ids in tqdm([('rn15/', range(1, 10)), ('rn30/', range(1, 18))]):
            subjects = {s_id: {} for s_id in subjects_ids}

            for s, labels in subjects.items():
                closed_path = self.path + rn + str(s) + '/closed/'
                opened_path = self.path + rn + str(s) + '/opened/'
                closed_imgs = listdir(closed_path)
                opened_imgs = listdir(opened_path)

                for closed in closed_imgs:
                    if 'right' in closed:
                        continue
                    number = int(closed.split('_')[0])
                    labels[number] = 1.0

                for opened in opened_imgs:
                    if 'right' in opened:
                        continue
                    number = int(opened.split('_')[0])
                    labels[number] = 0.0
            subjects_rn[rn] = subjects
        return subjects_rn


class Talking(Dataset):
    def __init__(self, folder_path):
        self.path = folder_path
        super(Talking, self).__init__('Talking')

    def load_pairs(self, input_size):
        in_r = []
        in_l = []
        y = []
        counts = [0, 0]

        for s in [1]:
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = listdir(closed_path)
            opened_imgs = listdir(opened_path)

            for closed in tqdm(closed_imgs):
                if 'right' in closed:
                    continue
                in_l.append(read_rgb_image(closed_path + closed, input_size, False))
                in_r.append(read_rgb_image(closed_path + closed.replace('left', 'right'), input_size, True))
                y.append(1.0)
                counts[1] += 1

            for opened in tqdm(opened_imgs):
                if 'right' in opened:
                    continue
                in_l.append(read_rgb_image(opened_path + opened, input_size, False))
                in_r.append(read_rgb_image(opened_path + opened.replace('left', 'right'), input_size, True))
                y.append(0.0)
                counts[0] += 1
        x = [np.array(in_r), np.array(in_l)]
        return x, y, counts

    def load_simple_two_classes(self, input_size):
        x = []
        y = []
        counts = [0, 0]
        for s in [1]:
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = listdir(closed_path)
            opened_imgs = listdir(opened_path)

            for closed in tqdm(closed_imgs):
                if 'right' in closed:
                    continue
                x.append(read_rgb_image(closed_path + closed, input_size, False))
                x.append(read_rgb_image(closed_path + closed.replace('left', 'right'), input_size, True))
                y.append([1.0, 0.0])
                y.append([1.0, 0.0])
                counts[1] += 1

            for opened in tqdm(opened_imgs):
                if 'right' in opened:
                    continue
                x.append(read_rgb_image(opened_path + opened, input_size, False))
                x.append(read_rgb_image(opened_path + opened.replace('left', 'right'), input_size, True))
                y.append([0.0, 1.0])
                y.append([0.0, 1.0])
                counts[0] += 1

        return np.array(x), np.array(y), counts

    def load_gts(self):
        subjects_ids = [1]
        subjects = {s_id: {} for s_id in subjects_ids}

        for s, labels in tqdm(subjects.items()):
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = listdir(closed_path)
            opened_imgs = listdir(opened_path)

            for closed in closed_imgs:
                if 'right' in closed or 'face' in closed:
                    continue
                number = int(closed.split('_')[0])
                labels[number] = 1.0

            for opened in opened_imgs:
                if 'right' in opened or 'face' in opened:
                    continue
                number = int(opened.split('_')[0])
                labels[number] = 0.0
        print(len(subjects[1]))
        return subjects


class EyeBlinking8(Dataset):
    def __init__(self, folder_path):
        self.path = folder_path
        super(EyeBlinking8, self).__init__('eyeblinking8')

    def load_pairs(self, input_size, subjects, undersample=False, random_subset=False):
        in_r = []
        in_l = []
        y = []
        counts = [0, 0]

        for s in tqdm(subjects):
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = [c for c in listdir(closed_path) if 'right' not in c]  # get left only, and load right later
            opened_imgs = [o for o in listdir(opened_path) if 'right' not in o]  # get left only, and load right later

            for closed in closed_imgs:
                try:
                    l_img, r_img = load_one_flipped_pair(closed_path + closed, closed_path + closed.replace('left', 'right'),
                                                         input_size)
                    in_l.append(l_img)
                    in_r.append(r_img)
                    y.append(1.0)
                except:
                    print('Failure loading image')

            if undersample:
                max_opened = len(closed_imgs)
                if len(opened_imgs) > max_opened:
                    opened_imgs = np.random.choice(opened_imgs, max_opened)

            for opened in opened_imgs:
                try:
                    l_img, r_img = load_one_flipped_pair(opened_path + opened, opened_path + opened.replace('left', 'right'),
                                                         input_size)
                    in_l.append(l_img)
                    in_r.append(r_img)
                    y.append(0.0)
                except:
                    print('Failure loading image')

        if random_subset == False:
            x = [np.array(in_r), np.array(in_l)]
        else:
            np.random.seed(42)
            num_samples = int(len(y) * random_subset)
            indx = np.random.choice(len(y), num_samples, False)
            x = [np.array(in_r)[indx], np.array(in_l)[indx]]
            y = np.array(y)[indx].tolist()

        for train_label in y:
            if train_label == 1.0:
                counts[1] += 1
            else:
                counts[0] += 1

        return x, y, counts

    def load_simple_two_classes(self, input_size, subjects, single_label=False):
        x = []
        y = []
        counts = [0, 0]
        for s in tqdm(subjects):
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = [c for c in listdir(closed_path) if 'right' not in c]  # get left only, and load right later
            opened_imgs = [o for o in listdir(opened_path) if 'right' not in o]  # get left only, and load right later

            for closed in closed_imgs:
                try:
                    l_img, r_img = load_one_flipped_pair(closed_path + closed, closed_path + closed.replace('left', 'right'),
                                                         input_size)
                    x.append(l_img)
                    x.append(r_img)
                    if single_label:
                        y.append(1.0)
                        y.append(1.0)
                    else:
                        y.append([1.0, 0.0])
                        y.append([1.0, 0.0])
                    counts[1] += 1
                except:
                    print('Failed to load image')

            for opened in opened_imgs:
                try:
                    l_img, r_img = load_one_flipped_pair(opened_path + opened, opened_path + opened.replace('left', 'right'), input_size)
                    x.append(l_img)
                    x.append(r_img)
                    if single_label:
                        y.append(0.0)
                        y.append(0.0)
                    else:
                        y.append([0.0, 1.0])
                        y.append([0.0, 1.0])
                    counts[0] += 1
                except:
                    print('Failed to load image')

        return np.array(x), np.array(y), counts

    def load_gts(self, subjects_ids):
        subjects = {s_id: {} for s_id in subjects_ids}

        for s, labels in tqdm(subjects.items()):
            closed_path = self.path + str(s) + '/closed/'
            opened_path = self.path + str(s) + '/opened/'
            closed_imgs = listdir(closed_path)
            opened_imgs = listdir(opened_path)

            for closed in closed_imgs:
                if 'right' in closed:
                    continue
                number = int(closed.split('_')[0])
                labels[number] = 1.0

            for opened in opened_imgs:
                if 'right' in opened:
                    continue
                number = int(opened.split('_')[0])
                labels[number] = 0.0

        return subjects


class RT_BENE(Dataset):
    def __init__(self, json_path, image_path):
        self.path = json_path
        self.image_path = image_path
        super(RT_BENE, self).__init__('rt_bene')

    def load_one_fold(self, fold_data):
        one_fold = {}
        for s in fold_data['subjects'].values():
            images = s['images']
            closed = [img for img, label in images.items() if label == 1.0]
            opened = [img for img, label in images.items() if label == 0.0]

            for im in closed:
                # TODO: remove name from json
                one_fold[s['path_left'].replace('/home/icub/Kevin/RT-GENE_dataset_eyes',
                                                self.image_path) + '/' + im] = 1.0

            for im in opened:
                # TODO: remove name from json
                one_fold[s['path_left'].replace('/home/icub/Kevin/RT-GENE_dataset_eyes',
                                                self.image_path) + '/' + im] = 0.0
        return one_fold

    def load_folds(self):
        folds = []
        validations = []
        with open(self.path) as json_file:  
            data = json.load(json_file)
            for f in data['kfold']:
                one_fold = self.load_one_fold(f)
                folds.append(one_fold)

            validation = self.load_one_fold(data['validation'])
            validations.append(validation)
        return folds, validations

    def load_pairs(self, input_size, training_folds, testing_folds, random_subset=False):
        train_y = []
        val_y = []
        counts = [0, 0]

        in_r = []
        in_l = []

        for f in tqdm(training_folds):
            for path, label in f.items():
                try:
                    l_img, r_img = load_one_flipped_pair(path, path.replace('left', 'right'), input_size)
                    in_l.append(l_img)
                    in_r.append(r_img)
                    train_y.append(label)
                except:
                    print('Failure loading image')

        if random_subset == False:
            train_x = [np.array(in_r), np.array(in_l)]
        else:
            np.random.seed(42)
            num_samples = int(len(train_y) * random_subset)
            indx = np.random.choice(len(train_y), num_samples, False)
            train_x = [np.array(in_r)[indx], np.array(in_l)[indx]]
            train_y = np.array(train_y)[indx].tolist()

        for train_label in train_y:
            if train_label == 1.0:
                counts[1] += 1
            else:
                counts[0] += 1

        in_r = []
        in_l = []

        for f in tqdm(testing_folds):
            for path, label in f.items():
                try:
                    l_img, r_img = load_one_flipped_pair(path, path.replace('left', 'right'), input_size)
                    in_l.append(l_img)
                    in_r.append(r_img)
                    val_y.append(label)
                except:
                    print('Failure loading image')

        val_x = [np.array(in_r), np.array(in_l)]

        return train_x, train_y, val_x, val_y, counts

    def load_simple_two_classes(self, input_size, training_folds, testing_folds, single_label=False):
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        counts = [0, 0]

        for f in training_folds:
            for path, label in f.items():
                try:
                    l_img, r_img = load_one_flipped_pair(path, path.replace('left', 'right'), input_size)
                    train_x.append(l_img)
                    train_x.append(r_img)

                    if label == 1.0:
                        counts[1] += 1
                        if single_label:
                            train_y.append(1.0)
                            train_y.append(1.0)
                        else:
                            train_y.append([label, 0.0])
                            train_y.append([label, 0.0])
                    else:
                        counts[0] += 1
                        if single_label:
                            train_y.append(0.0)
                            train_y.append(0.0)
                        else:
                            train_y.append([label, 1.0])
                            train_y.append([label, 1.0])
                except:
                    print('Failed to load image')

        for f in testing_folds:
            for path, label in f.items():
                try:
                    l_img, r_img = load_one_flipped_pair(path, path.replace('left', 'right'), input_size)
                    val_x.append(l_img)
                    val_x.append(r_img)
                    if label == 1.0:
                        if single_label:
                            val_y.append(1.0)
                            val_y.append(1.0)
                        else:
                            val_y.append([label, 0.0])
                            val_y.append([label, 0.0])
                    else:
                        if single_label:
                            val_y.append(0.0)
                            val_y.append(0.0)
                        else:
                            val_y.append([label, 1.0])
                            val_y.append([label, 1.0])
                except:
                    print('Failed to load image')

        return np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y), counts
