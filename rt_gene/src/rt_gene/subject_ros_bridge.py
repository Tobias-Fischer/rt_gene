"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from rt_gene.msg import MSG_SubjectImagesList, MSG_SubjectImages

from cv_bridge import CvBridge


class SubjectImages(object):
    def __init__(self, s_id):
        self.id = s_id
        self.face = None
        self.right = None
        self.left = None


class SubjectBridge(object):
    def __init__(self):
        self.__cv_bridge = CvBridge()

    def msg_to_images(self, subject_msg):
        subject = SubjectImages(subject_msg.subject_id)
        subject.face = self.__cv_bridge.imgmsg_to_cv2(subject_msg.face_img, "rgb8")
        subject.right = self.__cv_bridge.imgmsg_to_cv2(subject_msg.right_eye_img, "rgb8")
        subject.left = self.__cv_bridge.imgmsg_to_cv2(subject_msg.left_eye_img, "rgb8")
        return subject

    def images_to_msg(self, subject_id, subject):
        msg = MSG_SubjectImages()
        msg.subject_id = subject_id
        msg.face_img = self.__cv_bridge.cv2_to_imgmsg(subject.face_color, "rgb8")
        msg.right_eye_img = self.__cv_bridge.cv2_to_imgmsg(subject.right_eye_color, "rgb8")
        msg.left_eye_img = self.__cv_bridge.cv2_to_imgmsg(subject.left_eye_color, "rgb8")
        return msg


class SubjectListBridge(object):
    def __init__(self):
        self.__subject_bridge = SubjectBridge()

    def msg_to_images(self, subject_msg):
        subject_dict = dict()
        for s in subject_msg.subjects:
            subject_dict[s.subject_id] = self.__subject_bridge.msg_to_images(s)
        return subject_dict

    def images_to_msg(self, subject_dict, timestamp):
        msg = MSG_SubjectImagesList()
        msg.header.stamp = timestamp
        for subject_id, s in subject_dict.items():
            try:
                msg.subjects.append(self.__subject_bridge.images_to_msg(subject_id, s))
            except TypeError:
                pass

        return msg
