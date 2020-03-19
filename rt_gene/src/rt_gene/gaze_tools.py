"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np


def get_phi_theta_from_euler(euler_angles):
    return -euler_angles[2], -euler_angles[1]


def get_euler_from_phi_theta(phi, theta):
    return 0, -theta, -phi


def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def visualize_landmarks(image, landmarks):
    import cv2

    output_image = np.copy(image)
    for landmark in landmarks.reshape(-1, 2):
        cv2.circle(output_image, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
    return output_image


def limit_yaw(euler_angles_head):
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head


def crop_face_from_image(color_img, box):
    _bb = list(map(int, box))
    if _bb[0] < 0:
        _bb[0] = 0
    if _bb[1] < 0:
        _bb[1] = 0
    if _bb[2] > color_img.shape[1]:
        _bb[2] = color_img.shape[1]
    if _bb[3] > color_img.shape[0]:
        _bb[3] = color_img.shape[0]
    return color_img[_bb[1]: _bb[3], _bb[0]: _bb[2]]


def is_rotation_vector_stable(last_rotation_vector, current_rotation_vector):
    # check to see if rotation_vector is wild, if so, stop checking head positions
    _unit_rotation_vector = current_rotation_vector / np.linalg.norm(current_rotation_vector)
    _unit_last_rotation_vector = last_rotation_vector / np.linalg.norm(last_rotation_vector)
    _theta = np.arccos(np.dot(_unit_last_rotation_vector.reshape(3, ), _unit_rotation_vector))
    # tqdm.write("Head Rotation from last frame: {:.2f}".format(_theta))
    if _theta > 0.1:
        # we have too much rotation here, likely unstable, thus error out
        print('Could not estimate head pose due to instability of landmarks')
        return False
    else:
        return True


def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]

    return [left_x, top_y, right_x, bottom_y]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]

    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    return [left_x, top_y, right_x, bottom_y]


def angle_loss(y_true, y_pred):
    # noinspection PyUnresolvedReferences
    from tensorflow.keras import backend as K
    return K.sum(K.square(y_pred - y_true), axis=-1)


def accuracy_angle(y_true, y_pred):
    import tensorflow as tf

    pred_x = -1 * tf.cos(y_pred[0]) * tf.sin(y_pred[1])
    pred_y = -1 * tf.sin(y_pred[0])
    pred_z = -1 * tf.cos(y_pred[0]) * tf.cos(y_pred[1])
    pred_norm = tf.sqrt(pred_x * pred_x + pred_y * pred_y + pred_z * pred_z)

    true_x = -1 * tf.cos(y_true[0]) * tf.sin(y_true[1])
    true_y = -1 * tf.sin(y_true[0])
    true_z = -1 * tf.cos(y_true[0]) * tf.cos(y_true[1])
    true_norm = tf.sqrt(true_x * true_x + true_y * true_y + true_z * true_z)

    angle_value = (pred_x * true_x + pred_y * true_y + pred_z * true_z) / (true_norm * pred_norm)
    tf.clip_by_value(angle_value, -0.9999999999, 0.999999999)
    return (tf.acos(angle_value) * 180.0) / np.pi


def get_normalised_eye_landmarks(landmarks, box):
    eye_indices = np.array([36, 39, 42, 45])
    transformed_landmarks = landmarks[eye_indices]
    transformed_landmarks[:, 0] -= box[0]
    transformed_landmarks[:, 1] -= box[1]
    return transformed_landmarks
