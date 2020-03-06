import numpy as np

camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]

ros_to_camera = [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]


def position_ros_to_tf(ros_position):
    return np.array([ros_position.x, ros_position.y, ros_position.z])


def position_tf_to_ros(tf_position):
    from geometry_msgs.msg import Point
    return Point(tf_position[0], tf_position[1], tf_position[2])


def quaternion_ros_to_tf(ros_quaternion):
    return np.array([ros_quaternion.x, ros_quaternion.y, ros_quaternion.z, ros_quaternion.w])


def quaternion_tf_to_ros(tf_quaternion):
    from geometry_msgs.msg import Quaternion
    return Quaternion(tf_quaternion[0], tf_quaternion[1], tf_quaternion[2], tf_quaternion[3])


def geometry_to_tuple(geometry_msg):
    return geometry_msg.x, geometry_msg.y, geometry_msg.z


def convert_image(msg, desired_encoding="passthrough", ignore_invalid_depth=False):
    from cv_bridge import CvBridge

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
