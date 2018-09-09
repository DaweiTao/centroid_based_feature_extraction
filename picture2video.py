import cv2 as cv
import os
import glob

path_to_pics = './combined_output/'
video_name = "combined_real.avi"
fourcc = cv.VideoWriter_fourcc(*'XVID')
frame_width = 256
frame_height = 256


def create_dir_if_not_exists(dic):
    if not os.path.exists(dic):
        os.makedirs(dic)


frame_count = 0
image_list = glob.glob(path_to_pics + "*.png")
out = cv.VideoWriter(video_name, fourcc, 20.0, (frame_width, frame_height))

for img in image_list:
    frame = cv.imread(img)
    out.write(frame)
    frame_count += 1
    print("(" + str(frame_count) + " / " + str(len(image_list)) + ")")

    if cv.waitKey(30) & 0xff == ord('q'):
        break
