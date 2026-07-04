from cv_bridge import CvBridge
from rt_gene_interfaces.msg import SubjectImages, SubjectImagesArray


class SubjectImagesView:
    def __init__(self, subject_id):
        self.id = subject_id
        self.face = None
        self.right = None
        self.left = None


class SubjectBridge:
    def __init__(self):
        self._cv_bridge = CvBridge()

    def msg_to_images(self, msg):
        subject = SubjectImagesView(msg.subject_id)
        subject.face = self._cv_bridge.imgmsg_to_cv2(msg.face_image, "bgr8")
        subject.right = self._cv_bridge.imgmsg_to_cv2(msg.right_eye_image, "bgr8")
        subject.left = self._cv_bridge.imgmsg_to_cv2(msg.left_eye_image, "bgr8")
        return subject

    def images_to_msg(self, subject_id, subject):
        msg = SubjectImages()
        msg.subject_id = str(subject_id)
        msg.face_image = self._cv_bridge.cv2_to_imgmsg(subject.face_color, "bgr8")
        msg.right_eye_image = self._cv_bridge.cv2_to_imgmsg(subject.right_eye_color, "bgr8")
        msg.left_eye_image = self._cv_bridge.cv2_to_imgmsg(subject.left_eye_color, "bgr8")
        return msg


class SubjectArrayBridge:
    def __init__(self):
        self._subject_bridge = SubjectBridge()

    def msg_to_images(self, msg):
        return {subject.subject_id: self._subject_bridge.msg_to_images(subject) for subject in msg.subjects}

    def images_to_msg(self, subjects, header):
        msg = SubjectImagesArray()
        msg.header = header
        for subject_id, subject in subjects.items():
            if subject.left_eye_color is not None and subject.right_eye_color is not None:
                msg.subjects.append(self._subject_bridge.images_to_msg(subject_id, subject))
        return msg
