import numpy as np
from std_msgs.msg import Header

from rt_gene_ros.subject_bridge import SubjectArrayBridge


class Subject:
    def __init__(self):
        self.face_color = np.zeros((2, 2, 3), dtype=np.uint8)
        self.left_eye_color = np.zeros((2, 2, 3), dtype=np.uint8)
        self.right_eye_color = np.zeros((2, 2, 3), dtype=np.uint8)


def test_subject_image_headers_use_source_header():
    header = Header()
    header.stamp.sec = 12
    header.stamp.nanosec = 34
    header.frame_id = "camera_optical_frame"

    msg = SubjectArrayBridge().images_to_msg({0: Subject()}, header)
    subject = msg.subjects[0]

    assert msg.header == header
    assert subject.face_image.header == header
    assert subject.left_eye_image.header == header
    assert subject.right_eye_image.header == header
