#!/usr/bin/env python

"""
Convolutional Neural Network (CNN) for eye gaze estimation
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Hyung Jin Chang (hj.chang@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import os
import cv2
import time
from tqdm import tqdm

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospkg
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

# noinspection PyUnresolvedReferences
import rt_gene.gaze_tools as gaze_tools
from tf import TransformBroadcaster, TransformListener
import tf.transformations
import tensorflow
import collections

from rt_gene.subject_ros_bridge import SubjectListBridge
from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.gaze_tools import angle_loss, accuracy_angle


class GazeEstimator(object):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass (one per eye) of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation.
    Additionally, two images with the gaze overlaid on the eye images are published."""

    def __init__(self):
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))
        self.image_height = rospy.get_param("~image_height", 36)
        self.image_width = rospy.get_param("~image_width", 60)
        self.bridge = CvBridge()
        self.subjects_bridge = SubjectListBridge()

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()

        self.use_last_headpose = rospy.get_param("~use_last_headpose", True)
        self.tf_prefix = rospy.get_param("~tf_prefix", "gaze")
        self.last_phi_head, self.last_theta_head = None, None

        self.rgb_frame_id_ros = rospy.get_param("~rgb_frame_id_ros", "/kinect2_nonrotated_link")

        self.headpose_frame = self.tf_prefix + "/head_pose_estimated"
        self.device_id_gazeestimation = rospy.get_param("~device_id_gazeestimation", default="/gpu:0")
        with tensorflow.device(self.device_id_gazeestimation):
            config = tensorflow.ConfigProto(inter_op_parallelism_threads=1,
                                            intra_op_parallelism_threads=1)
            if "gpu" in self.device_id_gazeestimation:
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
            config.log_device_placement = False
            self.sess = tensorflow.Session(config=config)
            set_session(self.sess)

        model_files = rospy.get_param("~model_files")

        self.models = []
        for model_file in model_files:
            tqdm.write('Load model ' + model_file)
            model = load_model(os.path.join(rospkg.RosPack().get_path('rt_gene'), model_file),
                               custom_objects={'accuracy_angle': accuracy_angle, 'angle_loss': angle_loss})
            # noinspection PyProtectedMember
            model._make_predict_function()  # have to initialize before threading
            self.models.append(model)
        tqdm.write('Loaded ' + str(len(self.models)) + ' models')

        self.graph = tensorflow.get_default_graph()
        
        self.image_subscriber = rospy.Subscriber('/subjects/images', MSG_SubjectImagesList, self.image_callback, queue_size=3)
        self.subjects_gaze_img = rospy.Publisher('/subjects/gazeimages', Image, queue_size=3)
       
        self.average_weights = np.array([0.1, 0.125, 0.175, 0.2, 0.4])
        self.gaze_buffer_c = {}

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def estimate_gaze_twoeyes(self, test_input_left, test_input_right, headpose):
        test_headpose = headpose.reshape(1, 2)
        with self.graph.as_default():
            predictions = []
            for model in self.models:
                predictions.append(model.predict({'img_input_L': test_input_left,
                                                  'img_input_R': test_input_right,
                                                  'headpose_input': test_headpose})[0])
            mean_prediction = np.mean(np.array(predictions), axis=0)
            if len(self.models) == 1:  # only apply offset for single model, not for ensemble models
                mean_prediction[1] += 0.11
            return mean_prediction

    def visualize_eye_result(self, eye_image, est_gaze):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(eye_image)

        center_x = self.image_width / 2
        center_y = self.image_height / 2

        endpoint_x, endpoint_y = gaze_tools.get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
        return output_image

    def publish_image(self, image, image_publisher, timestamp):
        """This image publishes the `image` to the `image_publisher` with the given `timestamp`."""
        image_ros = self.bridge.cv2_to_imgmsg(image, "rgb8")
        image_ros.header.stamp = timestamp
        image_publisher.publish(image_ros)

    def input_from_image(self, eye_img_msg, flip=False):
        """This method converts an eye_img_msg provided by the landmark estimator, and converts it to a format
        suitable for the gaze network."""
        cv_image = eye_img_msg
        #cv_image = self.bridge.imgmsg_to_cv2(eye_img_msg, "rgb8")
        if flip:
            cv_image = cv2.flip(cv_image, 1)
        currimg = cv_image.reshape(self.image_height, self.image_width, 3, order='F')
        currimg = currimg.astype(np.float32)
        # print('currimg.dtype', currimg.dtype)
        # cv2.imwrite('/home/tobias/test_inplace.png', currimg)
        testimg = np.zeros((1, self.image_height, self.image_width, 3))
        testimg[0, :, :, 0] = currimg[:, :, 0] - 103.939
        testimg[0, :, :, 1] = currimg[:, :, 1] - 116.779
        testimg[0, :, :, 2] = currimg[:, :, 2] - 123.68
        return testimg
        
        
    def compute_eye_gaze_estimation(self, subject_id, timestamp,
                                    input_r,         input_l):
        """
        subject_id : integer,  id of the subject
        input_x    : cv_image, input image of x eye
        (phi_x)    : double,   phi angle estimated using pupil detection
        (theta_x)  : double,   theta angle estimated using pupil detection
        """
        try:
            lct = self.tf_listener.getLatestCommonTime(self.rgb_frame_id_ros, self.headpose_frame + str(subject_id))
            if (timestamp - lct).to_sec() < 0.25:
                tqdm.write('Time diff: ' + str((timestamp - lct).to_sec()))

                (trans_head, rot_head) = self.tf_listener.lookupTransform(self.rgb_frame_id_ros, self.headpose_frame + str(subject_id), lct)
                euler_angles_head = gaze_tools.get_head_pose(trans_head, rot_head)

                phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(euler_angles_head)
                self.last_phi_head, self.last_theta_head = phi_head, theta_head
            else:
                if self.use_last_headpose and self.last_phi_head is not None:
                    tqdm.write('Big time diff, use last known headpose! ' + str((timestamp - lct).to_sec()))
                    phi_head, theta_head = self.last_phi_head, self.last_theta_head
                else:
                    tqdm.write(
                        'Too big time diff for head pose, do not estimate gaze!' + str((timestamp - lct).to_sec()))
                    return

            est_gaze_c = self.estimate_gaze_twoeyes(input_l, input_r, np.array([theta_head, phi_head]))

            self.gaze_buffer_c[subject_id].append(est_gaze_c)           
            
            if len(self.average_weights) == len(self.gaze_buffer_c[subject_id]):
                est_gaze_c_med = np.average(np.array(self.gaze_buffer_c[subject_id]), axis=0, weights=self.average_weights)
                self.publish_gaze(est_gaze_c_med, timestamp, subject_id)
                tqdm.write('est_gaze_c: ' + str(est_gaze_c_med))
                return est_gaze_c_med

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.Exception) as tf_e:
            print(tf_e)
        except rospy.ROSException as ros_e:
            if str(ros_e) == "publish() to a closed topic":
                print("See ya")
        return None

    def image_callback(self, subject_image_list, masked_list=None):
        """This method is called whenever new input arrives. The input is first converted in a format suitable
        for the gaze estimation network (see :meth:`input_from_image`), then the gaze is estimated (see
        :meth:`estimate_gaze`. The estimated gaze is overlaid on the input image (see :meth:`visualize_eye_result`),
        and this image is published along with the estimated gaze vector (see :meth:`publish_image` and
        :func:`publish_gaze`)"""
        timestamp = subject_image_list.header.stamp
        subjects_gaze_img = None

        subjects_dict = self.subjects_bridge.msg_to_images(subject_image_list)
        for subject_id, s in subjects_dict.items():
            if subject_id not in self.gaze_buffer_c.keys():
                self.gaze_buffer_c[subject_id] = collections.deque(maxlen=5)
                
            input_r = self.input_from_image(s.right, flip=False)
            input_l = self.input_from_image(s.left, flip=False)
            gaze_est = self.compute_eye_gaze_estimation(subject_id, timestamp, input_r, input_l)
            
            if gaze_est is not None:
                r_gaze_img = self.visualize_eye_result(s.right, gaze_est)
                l_gaze_img = self.visualize_eye_result(s.left, gaze_est)
                s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)
                if subjects_gaze_img is None:
                    subjects_gaze_img = s_gaze_img
                else:
                    subjects_gaze_img = np.concatenate((subjects_gaze_img, s_gaze_img), axis=0)

        if subjects_gaze_img is not None:
            gaze_img_msg = self.bridge.cv2_to_imgmsg(subjects_gaze_img.astype(np.uint8), "bgr8")
            self.subjects_gaze_img.publish(gaze_img_msg)

    def publish_gaze(self, est_gaze, msg_stamp, subject_id):
        """Publish the gaze vector as a PointStamped."""
        theta_gaze = est_gaze[0]
        phi_gaze = est_gaze[1]
        euler_angle_gaze = gaze_tools.get_euler_from_phi_theta(phi_gaze, theta_gaze)
        quaternion_gaze = tf.transformations.quaternion_from_euler(*euler_angle_gaze)
        self.tf_broadcaster.sendTransform((0, 0, 0.05),  # publish it 5cm above the head pose's origin (nose tip)
                                          quaternion_gaze,
                                          msg_stamp,
                                          self.tf_prefix + "/world_gaze" + str(subject_id),
                                          self.headpose_frame + str(subject_id))


if __name__ == '__main__':
    try:
        rospy.init_node('estimate_gaze')
        gaze_estimator = GazeEstimator()
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        print("See ya")
    except rospy.ROSException as e:
        if str(e) == "publish() to a closed topic":
            print("See ya")
        else:
            raise e
    except KeyboardInterrupt:
        print("Shutting down")
