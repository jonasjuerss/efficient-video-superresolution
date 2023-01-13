import os
import random
import cv2
import numpy as np
import tensorflow as tf
"""
Based on: https://www.tensorflow.org/tutorials/load_data/video#create_frames_from_each_video_file
"""

def frames_from_video_file(folder_path_groundtruth: str, folder_path_generated: str,
                           n_frames: int, output_size):
    """
    Creates frames from each video file present for each category.

    Args:
      folder_path: File path to the scene folder
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
      Note that height and width here might vary
    """
    files = sorted([os.path.basename(f) for f in os.listdir(folder_path_groundtruth)])
    start = random.randint(0, len(files) - n_frames + 1)

    HR = []
    LR = []
    TECO = []
    for f in files[start: start + n_frames]:
        print(f)
        print(os.path.join(folder_path_groundtruth, f))
        img = cv2.imread(os.path.join(folder_path_groundtruth, f), 3).astype(np.float32)[:, : , ::-1]
        HR.append(img)
        icol_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        LR.append(icol_blur[::4, ::4, ::])
        img = cv2.imread(os.path.join(folder_path_generated, f), 3).astype(np.float32)[:, :, ::-1]
        TECO.append(img)

    HR = np.stack(HR, axis=0)  # [n_frames, height, width, channels]
    LR = np.stack(LR, axis=0)  # [n_frames, height // 4, width // 4, channels]
    TECO = np.stack(TECO, axis=0)  # [n_frames, height, width, channels]

    x1 = random.randint(0, LR.shape[1] - output_size[0] + 1)
    x2 = random.randint(0, LR.shape[2] - output_size[1] + 1)

    HR_cropped = HR[:, x1 * 4 : (x1 + output_size[0]) * 4, x2 * 4 : (x2 + output_size[1]) * 4, :]
    LR_cropped = LR[:, x1 : x1 + output_size[0], x2 : x2 + output_size[1], :]
    TECO_cropped = TECO[:, x1 * 4 : (x1 + output_size[0]) * 4, x2 * 4 : (x2 + output_size[1]) * 4, :]

    return (HR_cropped / 255, TECO_cropped / 255), LR_cropped / 255


class FrameGenerator:
    def __init__(self, path_groundtruth: str, path_generated: str, n_frames: int, output_size, training=False):
        """
        Returns a set of frames with their associated label.

        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
        """
        self.path_groundtruth = path_groundtruth
        self.path_generated = path_generated
        self.n_frames = n_frames
        self.output_size = output_size
        self.training = training
        self.scene_names = list(filter(lambda x: x != '', [os.path.basename(x[0]) for x in os.walk(path_groundtruth)]))

    def __call__(self):
        if self.training:
            random.shuffle(self.scene_names)

        for folder in self.scene_names:
            yield frames_from_video_file(os.path.join(self.path_groundtruth, folder),
                                         os.path.join(self.path_generated, folder),
                                         self.n_frames,
                                         self.output_size)
