import cv2
import numpy as np

def extract_image_one_fps(video_source_path):

    vidcap = cv2.VideoCapture(video_source_path)
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count * 1000)
        success, image = vidcap.read()

        ## Stop when last frame is identified
        image_last = cv2.imread("frame{}.png".format(count-1))
        if np.array_equal(image,image_last):
            break

        cv2.imwrite("input/frame%d.png" % count, image)
        print('{}.sec reading a new frame: {} '.format(count,success))
        count += 1

extract_image_one_fps(
    r'C:\Users\Pebaz\Videos\Captures\poc_game_language 2021-03-04 12-52-58.mp4'
)
