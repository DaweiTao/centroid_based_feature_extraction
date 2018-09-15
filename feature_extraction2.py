import cv2
import numpy as np
import glob
import os
import types

class FeatureExtractor2(object):

    def __init__(self, mode=0):
        self.image_count = 0
        self.mode = mode

    def get_color_component(self, rgb_img, color):
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

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        color_component = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        component_grayscale = cv2.cvtColor(color_component, cv2.COLOR_BGR2GRAY)
        _, component_binary = cv2.threshold(component_grayscale, 10, 255, cv2.THRESH_BINARY)

        # for blue binary image, we need to filp 0 & 1s
        if color == "B":
            component_binary = cv2.bitwise_not(component_binary)

        return color_component, component_binary

    def image_handler(self, img):
        red, red_binary = self.get_color_component(img, "R")
        # green, green_binary = self.get_color_component(img, "G")
        blue, blue_binary = self.get_color_component(img, "B")

        # dialation
        kernel = np.ones((2, 2), np.uint8)
        pupil_shape = cv2.dilate(red_binary, kernel, iterations=1)

        # find all eye sockets and pupil
        _, eye_sockets, hierarchy = cv2.findContours(blue_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, pupils, hierarchy = cv2.findContours(pupil_shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if no socket/pupil found, skip this img/frame
        if len(eye_sockets) == 0 or len(pupils) == 0:
            print("WARNING: pupil or eye_contour doesn't exist: " + str(self.image_count))
            self.image_count += 1
            return None, None, None

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

        msg = ""
        if leftmost[0] <= 0 or leftmost[0] >= 255 or leftmost[1] <= 0 or leftmost[1] >= 255:
            msg += " ,left eye corner touches boundary"
            leftmost = None

        if rightmost[0] <= 0 or rightmost[0] >= 255 or rightmost[1] <= 0 or rightmost[1] >= 255:
            msg += " ,right eye corner touches boundary"
            rightmost = None

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

        # calculate center/moment of the binary image of the pupil
        moment = cv2.moments(optimal_pupil_binary)
        x_centroid = int(moment["m10"] / moment["m00"])
        y_centroid = int(moment["m01"] / moment["m00"])

        # test if the pupil center is in the eye socket
        if self.mode == 1:
            ret = cv2.pointPolygonTest(optimal_eye_socket, (x_centroid, y_centroid), False)

            if ret < 0:
                print("WARNING: optimal pupil center is not in the optimal eye contour: " + str(self.image_count))
                self.image_count += 1
                return None, None, None

        print("Features extracted successfully: " + str(self.image_count) + msg)
        self.image_count += 1
        return leftmost, rightmost, (x_centroid, y_centroid)

    def extract(self, label_image, extract_pupil=True, extract_corner=True):
        if label_img.dtype != 'uint8':
            print("INPUT ERROR: dtype of input image is not unit8")
            return -1

        if label_img.shape != (256, 256, 3):
            print("INPUT ERROR: image shape should be (256, 256, 3)")
            return -1

        left_corner, right_corner, center = self.image_handler(label_image)

        if extract_pupil and extract_corner:
            return (center, left_corner, right_corner)

        if extract_pupil:
            return (center)

        if extract_corner:
            return (left_corner, right_corner)


if __name__ == '__main__':
    combined_path = "./combined_output"
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)

    fx2 = FeatureExtractor2()
    path_to_ml_label = './resources/test/'
    path_to_real_img = './resources/real/'
    test_list = glob.glob(path_to_ml_label + "*.png")
    real_list = glob.glob(path_to_real_img + "*.png")
    test_list = sorted(test_list, key=lambda x:x[-9:])
    real_list = sorted(real_list, key=lambda x:x[-9:])

    # print(len(label_list), len(image_list))
    min_len = len(test_list)
    if len(real_list) < min_len:
        min_len = len(real_list)

    for i in range(min_len):

        label_img_file = test_list[i]
        real_img_file = real_list[i]
#        print(label_img_file, real_img_file)

        label_img = cv2.imread(label_img_file)
        real_img = cv2.imread(real_img_file)
        coordinates = fx2.extract(label_img)

        if coordinates == -1:
            continue

        center = coordinates[0]
        left = coordinates[1]
        right = coordinates[2]

        if center is not None:
            cv2.circle(real_img, center, 3, (0, 255, 0), -1)

        if left is not None:
            cv2.circle(real_img, left, 3, (255, 0, 255), -1)

        if right is not None:
            cv2.circle(real_img, right, 3, (255, 0, 255), -1)

        cv2.imwrite("./combined_output/" + "{0:04d}".format(i) + ".png", real_img)
        # print("(" + str(i) + " / " + str(min_len) + ")")
