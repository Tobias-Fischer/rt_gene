"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import math
import numpy as np

from geometry_msgs.msg import Quaternion, Point
from cv_bridge import CvBridge
import tf.transformations


def position_ros_to_tf(ros_position):
    return np.array([ros_position.x, ros_position.y, ros_position.z])


def position_tf_to_ros(tf_position):
    return Point(tf_position[0], tf_position[1], tf_position[2])


def quaternion_ros_to_tf(ros_quaternion):
    return np.array([ros_quaternion.x, ros_quaternion.y, ros_quaternion.z, ros_quaternion.w])


def quaternion_tf_to_ros(tf_quaternion):
    return Quaternion(tf_quaternion[0], tf_quaternion[1], tf_quaternion[2], tf_quaternion[3])


def geometry_to_tuple(geometry_msg):
    return geometry_msg.x, geometry_msg.y, geometry_msg.z


def get_phi_theta_from_euler(euler_angles):
    return -euler_angles[2], -euler_angles[1]


def get_euler_from_phi_theta(phi, theta):
    return 0, -theta, -phi


def convert_image(msg, desired_encoding="passthrough", ignore_invalid_depth=False):
    type_as_str = str(type(msg))
    if type_as_str.find('sensor_msgs.msg._CompressedImage.CompressedImage') >= 0 \
            or type_as_str.find('_sensor_msgs__CompressedImage') >= 0:
        try:
            _, compr_type = msg.format.split(';')
        except ValueError:
            compr_type = ''
        if compr_type.strip() == 'tiff compressed':
            if ignore_invalid_depth:
                bridge = CvBridge()
                return bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
            else:
                raise Exception('tiff compressed is not supported')
        else:
            bridge = CvBridge()
            return bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
    else:
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)


def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def get_head_pose(trans_head, rot_head):
    euler_angles_head = list(tf.transformations.euler_from_quaternion(rot_head))
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head


def angle_loss(y_true, y_pred):
    from keras import backend as K
    return K.sum(K.square(y_pred - y_true), axis=-1)


def accuracy_angle(y_true, y_pred):
    from keras import backend as K
    import tensorflow

    pred_x = -1 * K.cos(y_pred[0]) * K.sin(y_pred[1])
    pred_y = -1 * K.sin(y_pred[0])
    pred_z = -1 * K.cos(y_pred[0]) * K.cos(y_pred[1])
    pred_norm = K.sqrt(pred_x * pred_x + pred_y * pred_y + pred_z * pred_z)

    true_x = -1 * K.cos(y_true[0]) * K.sin(y_true[1])
    true_y = -1 * K.sin(y_true[0])
    true_z = -1 * K.cos(y_true[0]) * K.cos(y_true[1])
    true_norm = K.sqrt(true_x * true_x + true_y * true_y + true_z * true_z)

    angle_value = (pred_x * true_x + pred_y * true_y + pred_z * true_z) / (true_norm * pred_norm)
    K.clip(angle_value, -0.9999999999, 0.999999999)
    return (tensorflow.acos(angle_value) * 180.0) / np.pi

