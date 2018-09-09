import cv2
import glob2
import numpy as np
import os

SOURCE_FILE_PATH = "./resources/test/"
FILE_FORMAT = "png"
image_index = 0


def make_test_dir():
    test_folders = ["R_component", "B_component", "feature_extraction", "optimal_pupil_mask", "abnormal"]
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

    # cv2.imwrite("./" + color + "_component/" +  "{0:04d}".format(image_count) + ".png", color_component)
    cv2.imwrite("./" + color + "_component/" + "{0:04d}".format(image_count) + ".png", component_binary)

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
    # cv2.imwrite("./pupil/" +  "{0:04d}".format(image_count) + ".png", pupil_shape)

    # find all eye contours and pupil
    _, eye_contours, hierarchy = cv2.findContours(blue_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, pupils, hierarchy = cv2.findContours(pupil_shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if no contour/pupil found, skip this img/frame
    if len(eye_contours) == 0 or len(pupils) == 0:
        print("WARNING: pupil or eye_contour doesn't exist: " + "{0:04d}".format(image_count))
        cv2.imwrite("./abnormal/" + "{0:04d}".format(image_count) + "_WN_NO" + ".png", img)
        image_count += 1
        return

    optimal_eye_contour_area = cv2.contourArea(eye_contours[0])
    optimal_eye_contour_index = 0

    for i in range(len(eye_contours)):
        area = cv2.contourArea(eye_contours[i])
        if area > optimal_eye_contour_area:
            optimal_eye_contour_index = i
            optimal_eye_contour_area = area

    cv2.drawContours(img, eye_contours, optimal_eye_contour_index, (255, 0, 255), 2)

    # extract left & right corner
    optimal_eye_contour = eye_contours[optimal_eye_contour_index]
    leftmost = tuple(optimal_eye_contour[optimal_eye_contour[:, :, 0].argmin()][0])
    rightmost = tuple(optimal_eye_contour[optimal_eye_contour[:, :, 0].argmax()][0])

    msg = ""

    if leftmost[0] <= 0 or leftmost[0] >= 255 or leftmost[1] <= 0 or leftmost[1] >= 256:
        msg += " ,left eye corner touches boundary"
        cv2.imwrite("./abnormal/" + "{0:04d}".format(image_count) + "_WN_LC" + ".png", img)
    else:
        cv2.circle(img, leftmost, 3, (0, 0, 0), -1)

    if rightmost[0] <= 0 or rightmost[0] >= 255 or rightmost[1] <= 0 or rightmost[1] >= 256:
        msg += " ,right eye corner touches boundary"
        cv2.imwrite("./abnormal/" + "{0:04d}".format(image_count) + "_WN_RC" + ".png", img)
    else:
        cv2.circle(img, rightmost, 3, (0, 0, 0), -1)

    # find optimal pupil
    optimal_pupil = cv2.contourArea(pupils[0])
    optimal_pupil_index = 0

    for i in range(len(pupils)):
        area = cv2.contourArea(pupils[i])
        if area > optimal_pupil:
            optimal_pupil_index = i
            optimal_pupil = area

    cv2.drawContours(img, pupils, optimal_pupil_index, (255, 133, 133), 2)

    # create mask for optimal pupil contour
    optimal_pupil_mask = np.zeros_like(img)
    cv2.drawContours(optimal_pupil_mask, pupils, optimal_pupil_index, (255, 255, 255), -1)
    optimal_pupil_mask = cv2.cvtColor(optimal_pupil_mask, cv2.COLOR_BGR2GRAY)
    _, optimal_pupil_binary = cv2.threshold(optimal_pupil_mask, 2, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./optimal_pupil_mask/" + "{0:04d}".format(image_count) + ".png", optimal_pupil_binary)

    # calculate center/moment of the binary image of the pupil
    moment = cv2.moments(optimal_pupil_binary)
    x_centroid = int(moment["m10"] / moment["m00"])
    y_centroid = int(moment["m01"] / moment["m00"])

    # test if the pupil center is in the eye contour
    ret = cv2.pointPolygonTest(optimal_eye_contour, (x_centroid, y_centroid), False)

    if ret < 0:
        print("WARNING: optimal pupil center is not in the optimal eye contour: " + "{0:04d}".format(image_count) + msg)
        cv2.imwrite("./abnormal/" + "{0:04d}".format(image_count) + "_WN_OPT" + ".png", img)
        image_count += 1
        return

    cv2.circle(img, (x_centroid, y_centroid), 2, (255, 255, 255), -1)
    msg += (", pupil center: " + str(x_centroid) + ", " + str(y_centroid))

    cv2.imwrite("./feature_extraction/" + "{0:04d}".format(image_count) + ".png", img)
    print("Features extracted successfully: " + "{0:04d}".format(image_count) + msg)
    image_count += 1


def video_handler():
    return


def feature_extraction_engine(path, source_type="image"):
    if source_type == "image":
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