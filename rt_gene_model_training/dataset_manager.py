#!/usr/bin/env python

import cv2
import numpy as np
from os import listdir

import json
import csv
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

class RT_BENE(object):
    def __init__(self, csv_subjects, input_size, random_subset):
        self.csv_subjects = csv_subjects
        self.input_size = input_size
        self.random_subset = random_subset
        self.subjects = {}
        self.folds = {} 

    '''
    |- root folder
        |- rt-bene_subjects.csv
        |- s000
            |- labels.csv
            |- img_0001.png
        |- s001
            |- labels.csv
        |- s002
            |- labels.csv
    '''

    def load_one_subject(self, csv_labels, left_folder, right_folder):
        subject = {}
        subject['y'] = []

        left_inputs = []
        right_inputs = []

        with open(csv_labels) as csvfile:
            csv_rows = csv.reader(csvfile)
            for row in tqdm(csv_rows):
                img_name = row[0]
                img_lbl = float(row[1])
                if img_lbl == 0.5:
                    continue
                left_img_path = left_folder + img_name
                right_img_path = right_folder + img_name.replace("left", "right")
                try:
                    left_img, right_img = load_one_flipped_pair(left_img_path, right_img_path, self.input_size)
                    left_inputs.append(left_img)
                    right_inputs.append(right_img)
                    subject['y'].append(img_lbl)
                except:
                    print('Failure loading pair!')

            if self.random_subset:
                np.random.seed(42)
                num_samples = int(len(subject['y']) * self.random_subset)
                indices = np.random.choice(len(subject['y']), num_samples, False)
                subject['x'] = [np.array(right_inputs)[indices], np.array(left_inputs)[indices]]
                subject['y'] = np.array(subject['y'])[indices].tolist()
            else:
                subject['x'] = [np.array(right_inputs), np.array(left_inputs)]
        return subject

    def load(self):
        self.folds = {}
        self.subjects = {}

        with open(self.csv_subjects) as csvfile:
            csv_rows = csv.reader(csvfile)
            for row in tqdm(csv_rows):
                subject_id = int(row[0])
                csv_labels = row[1]
                left_folder = row[2]
                right_folder = row[3]
                fold_type = row[4]
                fold_id = int(row[5])

                if fold_type == 'training':
                    print('\nsubject ' + str(subject_id) + ' is loading...')
                    csv_filename = self.csv_subjects.split('/')[-1]
                    csv_labels = self.csv_subjects.replace(csv_filename, csv_labels)
                    left_folder = self.csv_subjects.replace(csv_filename, left_folder)
                    right_folder = self.csv_subjects.replace(csv_filename, right_folder)
                    self.subjects[subject_id] = self.load_one_subject(csv_labels, left_folder, right_folder)
                    if fold_id not in self.folds.keys():
                        self.folds[fold_id] = []
                    self.folds[fold_id].append(subject_id)
                    
                elif fold_type == 'discarded':
                    print('\nsubject ' + str(subject_id) + ' is discarded.')
                    
                elif fold_type == 'validation':
                    print('\nsubject ' + str(subject_id) + ' is ignored (validation fold).')
                    
        print(self.folds)
        print(self.subjects.keys())

    def get_folds(self, fold_ids):
        fold = {}
        for fold_id in fold_ids:
            for subject_id in self.folds[fold_id]:
                # TODO: concat each subject x and y
                fold['x'] = 0
                fold['y'] = 0
        return fold

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
        
        
if __name__ == '__main__':
    csv_subjects = '/home/icub/Kevin/RT-GENE_dataset_eyes/subjects.csv'
    input_size = (96, 96)
    random_subset = None
    dataset = RT_BENE(csv_subjects, input_size, random_subset)
    
    dataset.load()
    
    
