import cv2
import glob2
import numpy as np
import os

SOURCE_FILE_PATH = "./resources/test/"
FILE_FORMAT = "png"
image_index = 0


def make_test_dir():
    test_folders = ["R_component", "B_component", "feature_extraction", "optimal_pupil_mask"]
    for dir in test_folders:
        test_dir = "./" + dir
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(test_dir + " --- directory created")
        else:
            print(test_dir + " --- directory exists")
    print("!!! Directory check complete !!!")
    print("--------------------------------\n")


def get_image_files():
    return glob2.glob(SOURCE_FILE_PATH + "**/*." + FILE_FORMAT)


def get_color_component(rgb_img, color):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    lower_hsv = ()
    upper_hsv = ()

    if color == "R":
        lower_hsv = (110, 50, 50)
        upper_hsv = (130, 255, 255)
    elif color == "G":
        lower_hsv = (40, 0, 0)
        upper_hsv = (80, 255, 255)
    elif color == "B":
        lower_hsv = (0, 50, 50)
        upper_hsv = (10, 255, 255)
    else:
        print("Invalid color component")
        return -1

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    color_component = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    component_grayscale = cv2.cvtColor(color_component, cv2.COLOR_BGR2GRAY)
    _, component_binary = cv2.threshold(component_grayscale, 10, 255, cv2.THRESH_BINARY)

    # for blue binary image, we need to filp 0 & 1s
    if color == "B":
        component_binary = cv2.bitwise_not(component_binary)

    # cv2.imwrite("./" + color + "_component/" + str(image_count) + ".png", color_component)
    cv2.imwrite("./" + color + "_component/" + str(image_count) + ".png", component_binary)

    return color_component, component_binary


image_count = 0


def image_handler(img):
    global image_count

    red, red_binary = get_color_component(img, "R")
    # green, green_binary = get_color_component(img, "G")
    blue, blue_binary = get_color_component(img, "B")

    # dialation
    kernel = np.ones((2, 2), np.uint8)
    pupil_shape = cv2.dilate(red_binary, kernel, iterations=1)
    # cv2.imwrite("./pupil/" + str(image_count) + ".png", pupil_shape)

    # find all eye sockets and pupil
    _, eye_sockets, hierarchy = cv2.findContours(blue_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, pupils, hierarchy = cv2.findContours(red_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if no socket/pupil found, skip this img/frame
    if len(eye_sockets) == 0 or len(pupils) == 0:
        print("Skip frame: pupil or eye_socket doesn't exist: " + str(image_count))
        image_count += 1
        return

    optimal_eye_socket_area = cv2.contourArea(eye_sockets[0])
    optimal_eye_socket_index = 0

    for i in range(len(eye_sockets)):
        area = cv2.contourArea(eye_sockets[i])
        if area > optimal_eye_socket_area:
            optimal_eye_socket_index = i
            optimal_eye_socket_area = area

    # extract left & right corner
    optimal_eye_socket = eye_sockets[optimal_eye_socket_index]
    leftmost = tuple(optimal_eye_socket[optimal_eye_socket[:, :, 0].argmin()][0])
    rightmost = tuple(optimal_eye_socket[optimal_eye_socket[:, :, 0].argmax()][0])
    cv2.drawContours(img, eye_sockets, optimal_eye_socket_index, (128, 0, 128), 2)

    msg = ""
    if leftmost[0] < 1 or leftmost[0] > 254 or leftmost[1] < 1 or leftmost[1] > 254:
        msg += " ,Missing left eye corner"
    else:
        cv2.circle(img, leftmost, 1, (0, 0, 0), 4)

    if rightmost[0] < 1 or rightmost[0] > 254 or rightmost[1] < 1 or rightmost[1] > 254:
        msg += " ,Missing right eye corner"
    else:
        cv2.circle(img, rightmost, 1, (0, 0, 0), 4)

    # find optimal pupil
    optimal_pupil = cv2.contourArea(pupils[0])
    optimal_pupil_index = 0

    for i in range(len(pupils)):
        area = cv2.contourArea(pupils[i])
        if area > optimal_pupil:
            optimal_pupil_index = i
            optimal_pupil = area

    optimal_pupil_mask = np.zeros_like(img)
    cv2.drawContours(optimal_pupil_mask, pupils, optimal_pupil_index, 255, -1)
    optimal_pupil_mask = cv2.cvtColor(optimal_pupil_mask, cv2.COLOR_BGR2GRAY)
    _, optimal_pupil_binary = cv2.threshold(optimal_pupil_mask, 2, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./optimal_pupil_mask/" + str(image_count) + ".png", optimal_pupil_binary)

    # calculate center/moment of the binary image of the pupil
    moment = cv2.moments(optimal_pupil_binary)
    x_centroid = int(moment["m10"] / moment["m00"])
    y_centroid = int(moment["m01"] / moment["m00"])
    cv2.circle(img, (x_centroid, y_centroid), 1, (0, 0, 0), 2)

    # test if the pupil center is in the eye socket
    ret = cv2.pointPolygonTest(optimal_eye_socket, (x_centroid, y_centroid), False)

    if ret < 0:
        print("Skip frame: two eyes in the frame, identified wrong pupil ")
        image_count += 1
        return

    cv2.imwrite("./feature_extraction/" + str(image_count) + ".png", img)
    print("Features extracted successfully: " + str(image_count) + msg)
    image_count += 1


def video_handler():
    return


def feature_extraction_engine(path, source_type="image"):
    if source_type ==  "image":
        image_handler(cv2.imread(path))
    elif source_type == "video":
        video_handler(cv2.VideoCapture(path))
    else:
        print("Invalid file format")


if __name__ == '__main__':
    make_test_dir()
    img_file_paths = get_image_files()

    # print(img_file_paths)
    for path in img_file_paths:
        feature_extraction_engine(path)

    print("\n!!! Task Complete !!!")
    print("--------------------------------\n")