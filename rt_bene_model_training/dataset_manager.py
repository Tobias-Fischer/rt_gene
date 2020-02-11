#!/usr/bin/env python

import cv2
import numpy as np
from os import listdir

import json
import csv
from tqdm import tqdm
import itertools


def read_rgb_image(img_path, size, flip):
    assert type(size) is tuple, "size parameter must be a tuple, (96, 96) for instance"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: can't read " + img_path)
    if flip:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, size, cv2.INTER_CUBIC)
    return img


def load_one_flipped_pair(l_path, r_path, size):
    l_img = read_rgb_image(l_path, size, False)
    r_img = read_rgb_image(r_path, size, True)
    return l_img, r_img

class RTBeneDataset(object):
    def __init__(self, csv_subjects, input_size):
        self.csv_subjects = csv_subjects
        self.input_size = input_size
        self.subjects = {}
        self.folds = {} 

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
            subject['x'] = [np.array(left_inputs), np.array(right_inputs)]
        return subject

    def load(self):
        self.folds = {}
        self.subjects = {}

        with open(self.csv_subjects) as csvfile:
            csv_rows = csv.reader(csvfile)
            for row in csv_rows:
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

    def get_folds(self, fold_ids):
        fold = {}
        subject_list = list(itertools.chain(*[self.folds[fold_id] for fold_id in fold_ids]))
        all_x_left = [self.subjects[subject_id]['x'][0] for subject_id in subject_list]
        all_x_right = [self.subjects[subject_id]['x'][1] for subject_id in subject_list]
        all_y = [np.array(self.subjects[subject_id]['y']) for subject_id in subject_list]
        fold['x'] = [np.concatenate(all_x_left), np.concatenate(all_x_right)]
        fold['y'] = np.concatenate(all_y)
        fold['positive'] = np.count_nonzero(fold['y']==1.)
        fold['negative'] = np.count_nonzero(fold['y']==0.)
        fold['y'] = fold['y'].tolist()
        return fold  
