import numpy as np
import cv2
import math
import copy as copy
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from time import sleep
from shapely.geometry import LineString
from shapely.geometry import Point
from numpy import ones, vstack
from numpy.linalg import lstsq
import time

import collections

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

test_flag = False

# Purpose: To find three green dots or three yellow dots and create an artificial circumference
# First function in the code will threshold the image based on color choice
# Seond Function will perform transformation to transform dots to points along circle speedometer

UPPER_GREEN = np.array([[140, 0, 215], [160, 11, 295]])
LOWER_GREEN = np.array([[39, 177, 139], [59, 197, 219]])

RGB_UPPER_GREEN = np.array([[70, 210, 150], [110, 255, 200]])
RGB_LOWER_GREEN = np.array([[70, 200, 80], [180, 255, 180]])

RGB_LOWER_GREEN2 = np.array([[135, 222, 55], [180, 255, 85]])

RGB_UPPER_GREEN2 = np.array([[220, 230, 2220], [255, 255, 255]])

# UPPER_GREEN = np.array([[1, 107, 48], [79, 154, 101]])
# LOWER_GREEN = np.array([[63, 190, 112], [153, 239, 138]])

# UPPER_GREEN = np.array([[79, 107, 48], [1, 154, 101]])
# LOWER_GREEN = np.array([[63, 239, 138], [153, 190, 112]])

# new_green = np.array([[80, 10, 24] , [140, 100, 80]])
new_green = np.array(
    [[(80 / 360) * 180, (25 / 100) * 255, (24 / 100) * 255], [(140 / 360) * 180, (100 / 100) * 255, (80 / 100) * 255]])
new_green2 = np.array(
    [[(77 / 360) * 180, (22 / 100) * 255, (21 / 100) * 255], [(142 / 360) * 180, (100 / 100) * 255, (83 / 100) * 255]])

greenLower = (39, 177, 139)
greenUpper = (64, 255, 255)


def removearray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def extend_line(slope, intercept, lefter_end_point, righter_end_point, extend_amount):
    non_vert_flag = True
    if lefter_end_point[0] < righter_end_point[0]:
        left_x = lefter_end_point[0]
        right_x = righter_end_point[0]
    elif lefter_end_point[0] > righter_end_point[0]:
        left_x = righter_end_point[0]
        right_x = lefter_end_point[0]
    else:
        non_vert_flag = False
        # if lefter_end_point[1] < righter_end_point[1]:
        #     left_point = list(lefter_end_point[0], lefter_end_point[1] - extend_amount)
        #     right_point = list(righter_end_point[0], righter_end_point[1] + extend_amount)
        # else:
        #     left_point = list(lefter_end_point[0], lefter_end_point[1] + extend_amount)
        #     right_point = list(righter_end_point[0], righter_end_point[1] - extend_amount)
        left_point = (lefter_end_point[0], (lefter_end_point[1] + extend_amount))
        right_point = (lefter_end_point[0], (lefter_end_point[1] - extend_amount))

    if non_vert_flag == True:

        left_x = left_x - extend_amount
        right_x = right_x + extend_amount

        left_point = (left_x, (slope * left_x) + intercept)
        right_point = (right_x, (slope * right_x) + intercept)
    else:
        left_point = (lefter_end_point[0], (lefter_end_point[1] + extend_amount))
        right_point = (lefter_end_point[0], (lefter_end_point[1] - extend_amount))

    return (left_point, right_point)


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def line_equation(point_one, point_two):
    points = [point_one, point_two]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return (m, c)


def getdot(frame):
    # camera = cv2.VideoCapture(0)
    while True:
        # grab the current frame
        # (grabbed, frame) = camera.read()

        # frame = cv2.imread("green3.jpg")

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        # if args.get("video") and not grabbed:
        #    break

        # resize the frame, blur it, and convert it to the HSV
        # color space

        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        output = frame.copy()
        height, width, depth = frame.shape
        out = np.zeros((height, width, 3), np.uint8)
        out[:] = (0, 0, 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # Upper and lower bound
        boundaries = UPPER_GREEN
        boundaries2 = LOWER_GREEN
        boundaries3 = new_green
        boundaries4 = new_green2
        # mask2 = cv2.inRange(hsv, boundaries[0], boundaries[1])
        # mask3 = cv2.inRange(hsv, boundaries2[0], boundaries2[1])
        mask4 = cv2.inRange(hsv, boundaries3[0], boundaries3[1])
        mask5 = cv2.inRange(hsv, boundaries4[0], boundaries4[1])

        mask_RGB = cv2.inRange(frame, RGB_LOWER_GREEN[0], RGB_LOWER_GREEN[1])
        mask_RGB2 = cv2.inRange(frame, RGB_UPPER_GREEN[0], RGB_UPPER_GREEN[1])
        mask_RGB3 = cv2.inRange(frame, RGB_LOWER_GREEN2[0], RGB_LOWER_GREEN2[1])

        mask = cv2.inRange(hsv, greenLower, greenUpper)
        # mask_total = (mask | mask2 | mask3 | mask4 | mask5)
        mask_total = (mask | mask4 | mask5)

        mask_total = cv2.bitwise_or(mask, mask4)
        mask_total = cv2.bitwise_or(mask_total, mask5)
        mask_total = cv2.bitwise_or(mask_total, mask_RGB)
        mask_total = cv2.bitwise_or(mask_total, mask_RGB2)
        mask_total = cv2.bitwise_or(mask_total, mask_RGB3)
        mask_total = cv2.erode(mask_total, None, iterations=2)
        mask_total = cv2.dilate(mask_total, None, iterations=2)

        # cv2.imshow("RGB", mask_RGB)
        # cv2.imshow("RGB2", mask_RGB2)

        # mask_total = cv2.cvtColor(mask_total, cv2.COLOR_BGR2GRAY)

        # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
        # mask_total = cv2.GaussianBlur(mask_total, (5, 5), 0);
        # mask_total = cv2.medianBlur(mask_total, 5)


        # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
        # mask_total = cv2.adaptiveThreshold(mask_total, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                             cv2.THRESH_BINARY, 11, 3.5)

        # kernel = np.ones((2.6, 2.7), np.uint8)
        # kernel = np.ones((2.6, 2.7))
        # mask_total = cv2.erode(mask_total, kernel, iterations=1)
        # mask_total = erosion

        # mask_total = cv2.dilate(mask_total, kernel, iterations=1)

        # cv2.imshow("Mask", mask_total)
        # cv2.waitKey(0)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask_total.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        i = 0
        circle_array = []
        # cv2.imshow("Mask", mask_total)
        # cv2.waitKey(0)

        # out = np.zeros_like(frame , dtype= np.uint8)
        # out = np.zeros_like(frame)
        # circle_array = []

        if len(cnts) >= 3:
            while i < 3:
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # only proceed if the radius meets a minimum size
                    if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(out, (int(x), int(y)), int(radius),
                                   (0, 255, 255), 20)
                        # cv2.circle(mask_total, center, 5, (0, 0, 255), -1)
                        cv2.circle(mask_total, center, int(radius),
                                   (0, 255, 0), 20)

                        # cv2.circle(out, center, 5, (0, 0, 255), -1)

                        # circle_array[i] = c
                        circle_array.append(c)
                        removearray(cnts, c)
                        i = i + 1
            # cv2.imshow("Mask", out)
            # cv2.waitKey(0)

            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

            conts = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
            # conts = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL,
            #                        cv2.CHAIN_APPROX_SIMPLE)[-2]
            conts = sorted(conts, key=cv2.contourArea, reverse=True)[:5]
            # cv2.imshow("Mask", frame)
            # cv2.waitKey(0)
            #return True, conts[0:3]
            return True, circle_array
        else:
            return False, -1


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "dot"
        shape_flag = False
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) >= 7:
            shape_flag = True
            return True
        else:
            return False


def fill_image(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contours, h = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    final = np.zeros(im.shape, np.uint8)
    mask = np.zeros(gray.shape, np.uint8)

    for i in range(0, len(contours)):
        mask[...] = 0
        cv2.drawContours(mask, contours, i, 255, -1)
        cv2.drawContours(final, contours, i, cv2.mean(im, mask), -1)

    # cv2.imshow('im', im)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    return final


# STACKEXCHANGE https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
def line_equation(point_one, point_two):
    if type(point_one) is tuple:
        points = [point_one, point_two]
        x_coords, y_coords = zip(*points)
        # x_coords, y_coords = zip(points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        return (m, c)
    else:
        return 1, 1


def extreme_points_add(point1_1, point1_2, point2_1, point2_2):
    smallest_x = 0
    smallest_y = 0
    largest_x = 0
    largest_y = 0

    point1_largest = [0, 0]
    point2_largest = [0, 0]

    point1_smallest = [0, 0]
    point2_smallest = [0, 0]

    if point1_1[0] >= point1_2[0]:
        point1_largest = point1_1
        point1_smallest = point1_2
    else:
        point1_largest = point1_2
        point1_smallest = point1_1
    if point2_1[0] >= point2_2[0]:
        point2_largest = point2_1
        point2_smallest = point2_2
    else:
        point2_largest = point2_2
        point2_smallest = point2_1

    if point1_smallest[0] < point2_smallest[0]:
        smallest_x = point1_smallest
    else:
        smallest_x = point2_smallest

    if point1_largest[0] > point2_largest[0]:
        largest_x = point1_largest
    else:
        largest_x = point2_largest

    return smallest_x, largest_x


def line_intersection_center(p0, p1, p2, p3):
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y

    if denom == 0: return None  # collinear

    denom_is_positive = denom > 0

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]

    s_numer = s10_x * s02_y - s10_y * s02_x

    if (s_numer < 0) == denom_is_positive: return None  # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x

    if (t_numer < 0) == denom_is_positive: return None  # no collision

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive: return None  # no collision

    # collision detected

    t = t_numer / denom

    intersection_point = [p0[0] + (t * s10_x), p0[1] + (t * s10_y)]

    return intersection_point


# THIS IS THE HELPER FUNCTION TO DELETE THESE LINES

def line_closest_to_mid(delete_lines, frame, center_point):
    height, width = frame.shape[:2]

    mid_point = width / 2

    index = 0

    offset = 2000

    current_index = 0

    for line, line2 in delete_lines:
        point1, point2 = line
        point3, point4 = line2
        dist1 = dist_formula((point1, point2), center_point)
        dist2 = dist_formula((point3, point4), center_point)
        if dist1 < offset:
            offset = dist1
            current_index = index
        if dist2 < offset:
            offset = dist2
            current_index = index
        index = index + 1

    return current_index


def delete_redundant_lines(lines, frame, center_point):
    # delete_these_line = lines[0]
    # delete_these_lines = np.delete(delete_these_line, 0, axis=0)
    delete_these_lines = []
    same_line_flag = False
    # lines_copy = np.copy(lines)
    for line in lines:
        line_slope1, line_intercept1 = line_equation(line[0], line[1])

        check_line = line
        slope_line = abs((line[0][1] - line[1][1]) / (line[0][0] - line[1][0]))
        if line in delete_these_lines:
            pass
        else:

            for current_line in lines:
                if type(current_line[0]) is tuple:
                    line_slope2, line_intercept2 = line_equation(current_line[0], current_line[1])
                    slope_current_line = abs(
                        (current_line[0][1] - current_line[1][1]) / (current_line[0][0] - current_line[1][0]))
                    if compare(current_line, line) or current_line in delete_these_lines:
                        pass

                    elif abs(line_slope1 - line_slope2) < .2:
                        if abs(line_intercept1 - line_intercept2) <= 20:
                            smallest_point, largest_point = extreme_points_add(current_line[0],
                                                                               current_line[1],
                                                                               line[0],
                                                                               line[1])
                            new_line = list(current_line)
                            new_line[0] = smallest_point
                            new_line[1] = largest_point

                            line = new_line
                else:
                    pass



                    # elif abs(line_slope1 - line_slope2) < .5:
                    #   if abs(line_intercept1 - line_intercept2) <= 30:
                    # SHOULD PROBABLY APPEND THESE LINES< SEEING AS THEY MAKE UP THE SUM TOTAL OF A DISTANCE
                    # delete_these_lines = np.append(delete_these_lines, current_line, axis=0)
                    #      delete_these_lines.append(line)
                    # same_line_flag = True

                    # if np.array_equal(current_line, lines[len(lines - 1)]) and same_line_flag == False:
                    # same_line_flag = True
                    # delete_these_lines = np.append(delete_these_lines, line, axis=0)
            # delete_these_lines = np.append(delete_these_lines, line, axis=0)
            if line not in delete_these_lines:
                if delete_these_lines:
                    check_flag = 0
                    for lines in delete_these_lines:
                        point = line[0]
                        point2 = line[1]
                        if (abs(point[0] - lines[0][0]) < 10 and abs(point[1] - lines[0][1]) < 10) or (
                                abs(point[0] - lines[1][0]) < 10 and abs(point[1] - lines[1][1]) < 10):
                            check_flag = 1
                        elif check_flag != 1:
                            delete_these_lines.append(line)

                else:
                    delete_these_lines.append(line)

    return [delete_these_lines[line_closest_to_mid(delete_these_lines, frame, center_point)]]


# First threshold the image
def smooth_edges(img):
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (1, 1), 0)
    edges = cv2.Canny(blur, 50, 130)
    # blur = cv2.GaussianBlur(edges, (5, 5), 0)
    return edges


def bounding_box_for_circles(c):
    # circle_points = []
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    circle_points = [extLeft, extRight, extTop, extBot]
    return circle_points


# SO Step1: Get HSV image
# STEP 2: Threshold the image to determine smooth_edges(
# Step3: Get circles
# STEP4: Draw a mask around those circles
# STEP%: Draw a circle at the radius of those points
# STEP 6: Get cOlor inside circles
# STEP 7: If circle has green or orange fill circle_radius(
# STEP 8:
def threshold_image(color, cv_image):
    boundaries_not = np.array([[0, 0, 0], [0, 0, 0]])
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    edges_mask = smooth_edges(cv_image)
    (cnts, _) = cv2.findContours(edges_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]
    screen_res = 1280, 720
    scale_width = screen_res[0] / cv_image.shape[1]
    scale_height = screen_res[1] / cv_image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(cv_image.shape[1] * scale)
    window_height = int(cv_image.shape[0] * scale)

    #    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    #    cv2.resizeWindow('dst_rt', window_width, window_height)
    #    cv2.imshow("dst_rt", edges_mask)
    # cv2.waitKey(0)

    shape = ShapeDetector()

    allcircles = []
    # should check for circles and get color boundaries
    for c in cnts:
        if len(c) > 5:
            extreme_values1 = bounding_box_for_circles(c)
            # if extreme_values1[1][0] < 2110 and extreme_values1[0][0] > 1400 and extreme_values1[2][1] > 1450 and extreme_values1[3][1] < 2110:
            #  j = 0

            if (extreme_values1[1][0] - extreme_values1[0][0]) > 200 and (
                        extreme_values1[3][1] - extreme_values1[2][1]) > 200:
                if shape.detect(c):
                    allcircles.append(c)

    screen_res = 1280, 720
    scale_width = screen_res[0] / cv_image.shape[1]
    scale_height = screen_res[1] / cv_image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(cv_image.shape[1] * scale)
    window_height = int(cv_image.shape[0] * scale)
    # cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Keypoints', window_height, window_width)

    cv2.drawContours(cv_image, allcircles, -1, (255, 255, 0), 3)

    # cv2.imshow("Keypoints", cv_image)
    # cv2.waitKey(0)

    if color == "green":
        # Upper and lower bound
        boundaries = UPPER_GREEN
        boundaries2 = LOWER_GREEN
    else:
        # ORANGE

        boundaries = np.array([5, 50, 50], np.uint8)
        boundaries2 = np.array([15, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, boundaries[0], boundaries[1])
    mask2 = cv2.inRange(hsv, boundaries2[0], boundaries2[1])
    total_mask = mask | mask2

    # Should just check for a color that we chose in this circle
    for c in allcircles:
        extreme_values = bounding_box_for_circles(c)
        # img[200:400, 100:300]
        crop_img = hsv[extreme_values[2][1]:extreme_values[3][1], (extreme_values[0][0]):(extreme_values[1][0])]
        mask3 = cv2.inRange(crop_img, boundaries[0], boundaries[1])
        mask4 = cv2.inRange(crop_img, boundaries2[0], boundaries2[1])
        total_mask2 = mask4 | mask3
        if np.count_nonzero(total_mask2) == 0:
            allcircles.remove(c)

    if len(allcircles) == 3:
        screen_res = 1280, 720
        scale_width = screen_res[0] / cv_image.shape[1]
        scale_height = screen_res[1] / cv_image.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(cv_image.shape[1] * scale)
        window_height = int(cv_image.shape[0] * scale)

        # cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Keypoints', window_width, window_height)

        cv2.drawContours(cv_image, allcircles, -1, (255, 255, 0), 3)
        # cv2.imshow("Keypoints", cv_image)
        # cv2.waitKey(0)
        return allcircles

    output = cv2.bitwise_and(cv_image, cv_image, mask=total_mask)
    output_light = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    output_upper = cv2.bitwise_and(cv_image, cv_image, mask=mask2)

    new_output = fill_image(output)

    screen_res = 1280, 720
    scale_width = screen_res[0] / cv_image.shape[1]
    scale_height = screen_res[1] / cv_image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(cv_image.shape[1] * scale)
    window_height = int(cv_image.shape[0] * scale)

    # cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst_rt', window_width, window_height)
    # cv2.namedWindow('dst_rt2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst_rt2', window_width, window_height)
    # cv2.namedWindow('dst_rt3', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst_rt3', window_width, window_height)

    # cv2.imshow("dst_rt", output)
    # cv2.imshow("dst_rt2", output_light)
    # cv2.imshow("dst_rt3", output_upper)
    # cv2.imshow("Original", cv_image)
    # cv2.waitKey(0)

    return output


def circle_center(dot_center):
    circle_array = []
    for circle in dot_center:
        M = cv2.moments(circle)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        circle_array.append((cX, cY))

    return circle_array


# DOt perimeter 1 should be the lowest point so we can reduce error possibility
def circle_radius(radial_point1, radial_point2, radial_center):
    radius = (
        math.sqrt(
            math.pow((radial_center[0] - radial_point1[0]), 2) + math.pow((radial_center[1] - radial_point1[1]), 2)))

    circumference = ((radius * 2) * math.pi)

    # Next part is point difference

    chord_length = dist_formula(radial_point1, radial_point2)
    radius1 = dist_formula(radial_center, radial_point1)
    radius2 = dist_formula(radial_center, radial_point2)

    approx_radius = radius1 + radius2

    approx_radius = approx_radius / 2

    # Equation from: Distance Formula

    point_distance = math.sqrt(
        math.pow((radial_point2[0] - radial_point1[0]), 2) + math.pow((radial_point2[1] - radial_point1[1]), 2))

    # Equation from: https://math.stackexchange.com/questions/185829/how-do-you-find-an-angle-between-two-points-on-the-edge-of-a-circle
    # theta_determinence is misleading, we actually only need the value in radiaans
    if radius == 0:
        return False

    theta_determinence = (((2 * math.pow((radius), 2)) - math.pow(point_distance, 2)) / (2 * (math.pow((radius), 2))))
    theta_determinence = math.acos(theta_determinence)

    arc_length = radius * theta_determinence

    return True, arc_length, approx_radius, radial_point1, radial_point2, radial_center


# THe distance formula since we're using it so fucking much
def dist_formula(point1, point2):
    point_distance = (math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow((point1[1] - point2[1]), 2)))
    return point_distance


def normalized_line(frame, tip_point, center_point):
    frame_copy = frame
    height, width, depth = frame_copy.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.line(mask, tip_point, center_point, (0, 0, 255), 1)


# Purpose: To establish the percentiles that an arc_length represents of a full circle
# need arc length and radius to estamate the length/circumference percentage
def arc_percentage(arc_length1, radius):
    percentage = arc_length1 / (2 * math.pi * radius) * 100
    return (percentage)


# Purpose: To return the differnce between tow arc lengths
def percentile_difference(arc_length1, arc_length2, radius):
    percentage_value = arc_percentage(arc_length1, radius)
    percentage_max = arc_percentage(arc_length2, radius)

    return ((percentage_value / percentage_max))


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def findArrow(img, dot_center, dot_center_radius, minLen):
    # uncorrected_dot_center = uncorrect_y_axis(dot_center, height)
    # crop_img = img[uncorrected_dot_center[1] - 80:uncorrected_dot_center[1] + 80,
    #          uncorrected_dot_center[0] - 100:uncorrected_dot_center[0] + 100]


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, s, v = cv2.split(hsv_image)

    # PYimageSrach http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    sigma = .05
    image_median = np.median(gaussian_blur)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * image_median))
    upper = int(min(255, (1.0 + sigma) * image_median))

    # edges = cv2.Canny(gaussian_blur, lower, upper)
    sobelx = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=5)  # y

    laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_64F)
    abs_lap64f = np.absolute(laplacian)
    lab_8u = np.uint8(abs_lap64f)
    edges = auto_canny(gaussian_blur, sigma)
    edges2 = auto_canny(hue, sigma)

    # edges2 = auto_canny(laplacian, sigma)

    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    mask_total = cv2.erode(edges, None, iterations=2)
    mask_total = cv2.dilate(edges, None, iterations=2)
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Edges', window_width, window_height)
    # cv2.imshow("Edges", edges)
    # #13 = {ndarray}[[289 152 392 141]]
    # cv2.waitKey(0)

    (height2, width2) = img.shape[:2]
    if minLen:
        pass
    else:
        minLen = 100

    # maxlineGap wasa 10
    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=66, minLineLength=80, maxLineGap=5)# LINE GAP WAS 10 when it worked
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=66,
                            minLineLength=minLen, maxLineGap=10)
    lines_copy = lines
    minLen = minLen + 5
    while len(lines) > 3:
        lines_copy = lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=66,
                                minLineLength=minLen, maxLineGap=10)
        minLen = minLen + 5
    lines = lines_copy
    minLen = minLen -5

    if len(lines) == 0:
        print("BREAK")
        return False

    # Here is where we delete lines that do not intersect the circledot
    i = 0
    new_lines = []
    extLeft = [dot_center[0] - dot_center_radius, dot_center[1] + dot_center_radius]
    extRight = [dot_center[0] - dot_center_radius, dot_center[1] - dot_center_radius]
    extBot = [dot_center[0] + dot_center_radius, dot_center[1] - dot_center_radius]
    extTop = [dot_center[0] + dot_center_radius, dot_center[1] + dot_center_radius]
    for line in lines:
        new_line0 = (line[0][0], line[0][1])  # correct_y_axis((line[0][0], line[0][1]), height)
        new_line1 = (line[0][2], line[0][3])  # correct_y_axis((line[0][2], line[0][3]), height)
        m, c = line_equation(new_line0, new_line1)
        total_new_line = extend_line(m, c, new_line0, new_line1, (width2 * .05))
        point1 = line_intersection_center(total_new_line[0], total_new_line[1], extLeft, extRight)
        point2 = line_intersection_center(total_new_line[0], total_new_line[1], extLeft, extTop)
        point3 = line_intersection_center(total_new_line[0], total_new_line[1], extRight, extBot)
        point4 = line_intersection_center(total_new_line[0], total_new_line[1], extBot, extTop)

        if point1 or point2 or point3 or point4:
            new_lines.append((new_line0, new_line1))

    if new_lines:
        lines = delete_redundant_lines(new_lines, img, dot_center)
    else:
        return False
    if len(lines) > 1:
        # Actually best way is to determine which is closer to houghlines bound being lower for gap
        # So create variable line_gap min and keep lowering it by 2 until its 1 at which point just pick the second one
        return False

    perimeter = []
    max_value = -1
    line_tip = (0, 0)
    center_line = (0, 0)
    max_line = [(0, 0, 0,)]
    slope_of_max = 0
    for line in lines:
        p1 = line[0]
        p2 = line[1]
        x1 = p1[0]
        y1 = p1[1]

        x2 = p2[0]
        y2 = p2[1]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', window_width, window_height)
        # cv2.imshow("img", img)
        #
        # cv2.waitKey(0)

        assume_center = [x1, y1]
        assume_tip = [x2, y2]

        # This normalizes the value so that we know the proper endpoints and their orientation relative to our "origin" AKA Dot Center

        if (abs(dist_formula(dot_center, assume_center)) >= abs(dist_formula(dot_center, assume_tip))):
            line_tip = (x1, y1)
            center_line = (x2, y2)
        else:
            line_tip = (x2, y2)
            center_line = (x1, y1)

        # since Im bad this is meant ot unpack the dot from a list of coordinates, whicha re incidentally stored in a tuple
        true_dot_center = dot_center

        slope = ((true_dot_center[1] - line_tip[1]) / (line_tip[0] - true_dot_center[0]))

        line_length = dist_formula(line_tip, center_line)

        # So one approach to reduce the error of the dots is to continuosly refine the slope by taking the closest poitn to the y-axis and averagin that with the "dot center"
        # Do this with the 10 closest points on the line we have coputed and extrapolated
        # Eventually what this will do is create a dot center that has been averaged to fit the actual center of the speedometer
        # from this we can constantly refine our radius and the error percentage of our speedometer guesses.

        line_value = [(line_tip, center_line, line_length)]

        if abs(line_length) > max_value:
            max_value = abs(line_length)
            max_line = line_value
            slope_of_max = slope
    return True, max_line, slope_of_max, minLen


# Purpose: to setermine the point that is a given distance a way from another
# TAken From here: https://math.stackexchange.com/questions/164541/finding-a-point-having-the-radius-chord-length-and-another-point
# THe idea is that A will be on the 0 degree a xis so any distance theta is just the distance between the points
# NOTE: Coordinates should be int because we are approximating an aexact pixel


# FORMULA
# β=α−θ, where (as shown below) θ=2asind2r, to get the location of BB.
def point_approximation(precision, radius, base_angle):
    theta_value = 2 * math.asin(precision / (2 * radius))

    beta_angle = base_angle - theta_value

    x_coord = math.cos((beta_angle)) * radius
    y_coord = math.sin((beta_angle)) * radius

    new_coords = (int(x_coord), int(y_coord))
    # 1 For loops greater than 3.14, for loops in the negative range less than 3.1415
    # if((abs(base_angle) > math.pi and base_angle > 0) or (abs(base_angle) < math.pi and base_angle < 0)):
    return new_coords
    # return new_coords, (base_angle - theta_value)


# The subtractions are done like this do to the nature of normalizing the axis and origin such that the point_cneter acts as the new origin
def find_angle(point_center, point_perimeter):
    x_dist = point_perimeter[0] - point_center[0]
    y_dist = point_center[1] - point_perimeter[1]

    angle_def = math.atan(((-1 * y_dist) / x_dist))
    if (x_dist < 0 and y_dist < 0):
        angle_def = (-1 * angle_def) - math.pi

    return (angle_def)



    # return(max_line)


# Purpose: To get an array that estimates the points along a semicircle given a radius, precision and start/end points
# Use Case: To find intersenction between speedometer line and circular boundary
# Not Lowest Precision is likely sqrt 2 due to the general nature of circles and pixels
# Parameters: Need radisu to determine new points along radisu, precision because we need to get a certain number of points along circle
# Start and end duh , and distance because we're doing a Euclidean distance between points


# get center point and normalize the points

# Not the most precise, but we can get a good APproximation of lengths
def circular_approx(radius, precision, base_angle, end_point, between_distance, arc_distance):
    distance_travesered = precision
    coordinate_location = []
    while distance_travesered <= between_distance:
        # Could potentially fix the less than 0 issue by artificially making the base angle less than base angle, perhaps 30 % distance of end_distance
        coordinate_location.append(point_approximation(distance_travesered, radius, base_angle))
        # base_angle = point_approximation(distance_travesered, radius, base_angle)[1]
        # base_angle = point_approximation(distance_travesered, radius, base_angle)[1]
        distance_travesered = distance_travesered + precision
    return coordinate_location


def find_intersection2(radius_point, lower_point, upper_point, slope, radius, height):
    new_radius = correct_y_axis(radius_point, height)
    new_lower_point = correct_y_axis(lower_point, height)
    # new_upper_point = correct_y_axis(upper_point, height)
    # slope = -1* slope

    p = Point(radius_point[0], radius_point[1])
    c = p.buffer(radius).boundary

    l1 = LineString([(radius_point), ((radius_point[0] + radius + 1), (radius_point[1] + ((radius + 1) * slope)))])
    l2 = LineString([(radius_point), ((radius_point[0] - (radius + 40)), (radius_point[1] + ((radius + 40) * slope)))])

    intersect = c.intersection(l2)

    if not intersect:
        return False, (0,0)

    lower_x = 0
    greater_x = 1

    if lower_point[0] <= upper_point[0]:
        lower_x = lower_point[0]
        greater_x = upper_point[0]
    else:
        greater_x = lower_point[0]
        lower_x = upper_point[0]

    intersect_list = np.array(intersect)

    if intersect_list[0] >= lower_x and intersect_list[0] <= greater_x:
        if intersect_list[1] <= lower_point[1] and intersect_list[1] >= upper_point[1]:
            return True, intersect_list
        else:
            # THIS IS FOR THE CASE THAT WE A RE ACTUALLY AT 0 or not moving
            return True, (-1, 2)
    elif intersect_list[0] <= lower_x and intersect_list[1] <= lower_point[1] and intersect_list[1] >= upper_point[1]:
        return True, intersect_list
    elif intersect_list[1] <= upper_point[1] and intersect_list[0] >= lower_x and intersect_list[0] <= greater_x:
        return True, intersect_list

    else:
        return True, (-1, -1)


def find_intersection(x_value, y_value, arcLength_values, slope):
    # intersection = list(set(dot_center) & set(arrow_array))

    # If there is no interesection we need to clculate slope of line that goes from tip of arrow to the dot_center
    # For now we will just assume that the arrow is represented by the leftmost point
    # Ill use a bounding box of the arc values and the coordinate circle

    # NEW IDEA: PASS the radius of a circle, then check for interesection points on the full circle
    # From here check if intersection on given ar clength by checking ifits in the  "arc-length box"
    # If so calculate the percentage of total arclength, by subtracting it from the whole and subtracting this result from the whole
    # take this get a percentage by dividing it by the whole, and voila you multiply that percent to the max pseed to get your answer

    boundaries_x = -1 * x_value
    boundaries_y = -1 * y_value

    p = Point(5, 5)
    c = p.buffer(3).boundary
    l = LineString([(0, 0), (10, 10)])
    i = c.intersection(l)

    while x_value > boundaries_x and y_value > boundaries_y:

        current_point = (x_value, y_value)
        for (x, y) in arcLength_values:
            # We're setting a bound condition
            if (current_point[0] >= (x - 3)) and (current_point[0] <= (x + 3)):
                if (current_point[1] >= (y - 3)) and (current_point[1] <= (y + 3)):
                    intersection_point = (x, y)
                    return (intersection_point)
        x_value = x_value - 1
        y_value = y_value - slope

    return (-1, -1)


def correct_y_axis(point, height):
    new_point = [point[0], point[1]]
    new_point[1] = (-1 * point[1]) + height
    return new_point


def uncorrect_y_axis(point, height):
    new_point = [point[0], point[1]]
    new_point[1] = (-1 * point[1]) + height
    new_point[1] = (point[1] - height) * -1
    return new_point


def correct_near_circular_points(point_outer, point_radius, new_radius, height):
    old_line = [point_radius, point_outer]

    p = Point(point_radius[0], point_radius[1])
    c = p.buffer(new_radius).boundary
    left_x = 0
    right_x = 0
    slope = 0
    new_point_outer = list(((0, 0), (0, 0)))
    new_point_radius = list(((0, 0), (0, 0)))

    if point_outer[0] < point_radius[0]:
        left_x = point_outer
        right_x = point_radius
        slope = (right_x[1] - left_x[1]) / (right_x[0] - left_x[0])
        intercept = point_outer[1] - slope * point_outer[0]
        new_line = extend_line(slope, intercept, left_x, right_x, height)
        new_point_outer = new_line[0]
        new_point_radius = new_line[1]
    elif point_outer[0] > point_radius[0]:
        right_x = point_outer
        left_x = point_radius
        slope = (right_x[1] - left_x[1]) / (right_x[0] - left_x[0])
        intercept = point_radius[1] - slope * point_radius[0]
        # slope, intercept, lefter_end_point, righter_end_point, extend_amount
        new_line = extend_line(slope, intercept, left_x, right_x, 30)
        new_point_outer = new_line[0]
        new_point_radius = new_line[1]
    else:
        if point_outer[1] < point_radius[1]:
            new_point_outer = [point_outer[0], (point_outer[1] + 30)]
            new_point_radius = [point_outer[0], (point_outer[1] - 30)]
        else:
            new_point_outer = [point_outer[0], (point_outer[1] - 30)]
            new_point_radius = [point_outer[0], (point_outer[1] + 30)]

    # slope = (right_x[1] - left_x[1]) / (right_x[0] - left_x[0])

    l1 = LineString([new_point_radius, new_point_outer])
    # l2 = LineString([(point_radius), ((point_radius[0] - radius + 1), (point_radius[1] + ((radius + 1) * slope)))])


    intersect = c.intersection(l1)

    intersection_array = np.array(intersect)

    intersection_list = intersection_array.tolist()
    flatten_list = intersection_array.flatten().tolist()
    shortest = 1000000
    iterate = 0
    index = 0
    if (len(flatten_list) > 2):
        for pointer in intersection_list:
            if len(pointer) == 1:
                break
            new_shortest = dist_formula(point_outer, pointer)
            if new_shortest < shortest:
                shortest = new_shortest
                index = iterate
            iterate = iterate + 1
        return intersection_list[index]
    else:
        return intersection_array


def Speedometer_Calibration(max_speed, frame):
    distance = 0
    radius = 0
    point_center = 0
    point1 = 0
    point2 = 0

    arcLength_values = []
    # cap = cv2.VideoCapture(camera_name)
    # ret, frame = cap.read()

    # blackened_image = threshold_image("green", frame)

    # TEST1-> determine types of green
    # frame = cv2.imread('speedometer3.jpg')
    frame = imutils.resize(frame, width=600)
    (height, width) = frame.shape[:2]

    circles_flag, circles = getdot(frame)
    true_circles = circles

    # circle_detector= ShapeDetector()
    #
    # for c in circles:
    #     if circle_detector.detect(c) is True:
    #         true_circles.append(c)
    if not circles_flag:
        return False
    else:
        if len(circles) != 3:
            return False
        circles_copy = list(circles)

        center_points = circle_center(circles_copy)
        # circles_copy = copy.deepcopy(circles)
        i = 0

        lower_circle = find_zero_circle(center_points)
        center_points.remove(lower_circle)
        center_points.append(lower_circle)
        triangle_flag, index_of_center = find_triangle(center_points, 2)
        if triangle_flag == False:
            return False
        right_circle = index_of_center
        # central_circle = center_points[index_of_center]
        center_points.remove(right_circle)
        upper_circle = center_points[0]

        new_flag, distance, radius, point1, point2, point_center = circle_radius(lower_circle, upper_circle,
                                                                                 right_circle)

        if new_flag == False:
            return False

        # ((x, y), radius_green_dot) = cv2.minEnclosingCircle(central_circle)
        radius_green_dot = int(radius / 3)

        if point1[1] > point2[1]:
            zero_point = point1
            max_point = point2
        else:
            zero_point = point2
            max_point = point1

        new_zero_point = zero_point  # correct_y_axis(zero_point, height)
        new_point_center = point_center  # correct_y_axis(point_center, height)
        new_max_point = max_point  # correct_y_axis(max_point, height)

        new_zero_point = correct_near_circular_points(zero_point, point_center, radius, height)
        new_max_point = correct_near_circular_points(max_point, point_center, radius, height)

        base_angle = find_angle(new_point_center, new_zero_point)

        # This line is likely superfluous, as we do not ever need to know the linear distance between our two endpoints
        # Only arc length and proportions

        dist_between = dist_formula(point1, point2)

        # Why have I chose the sqare root of 2: I believ it has to do with the hypotenuse between two diagonal pixels of 1 pixel length
        # Note I have changed my initial algoritihm to be based around arc_Lengtha

        # just remember you wont have a perfect circle between points, but you can approximate

        #arcLength_values = circular_approx(radius, 1, base_angle, new_max_point, dist_between, distance)
        arcLength_values = 0
        x_value = point_center[0]
        y_value = point_center[1]
        return (True, new_zero_point, new_max_point, arcLength_values, new_point_center, radius, radius_green_dot)


def Speedometer_get_Speed(point1, point2, height, frame, point_center, max_speed, radius, radius_green_dot, minLen):
    # radius_point, upper_point, lower_point, slope, radius, height
    if minLen:
        pass
    else:
        minLen = 95

    slope = 0



    complete_flag, speed_arrow, slope, minLen = findArrow(frame, point_center, radius_green_dot, minLen)
    if complete_flag == False:
        return 0, 100
    intersection_point = (0, 0)

    # Remember the arlength_values are relative to the "origin" which is the value of X-value and y-Value

    complete_flag, intersection_point = find_intersection2(point_center, point1, point2, slope, radius, height)

    if complete_flag == False:
        return 0, minLen
    if isinstance(intersection_point, tuple):
        if (-1, 2) == intersection_point:
            return 0, minLen
    else:
        pass

    if (not compare(intersection_point, (-1, -1))):
        new_flag, arc_length_intersection, approx_radius2, radial_point21, radial_point22, radial_center2 = circle_radius(
            point1, intersection_point, point_center)
        if new_flag == False:
            return 0, minLen
        new_flag, arc_length_max, approx_radius3, radial_point13, radial_point23, radial_center3 = circle_radius(point1,
                                                                                                                 point2,
                                                                                                                 point_center)
        if new_flag == False:
            return 0, minLen
        true_speed = percentile_difference(arc_length_intersection, arc_length_max, radius) * max_speed
        return true_speed, minLen
    if compare(intersection_point, (-1, -1)):

        if slope > ((point_center[1] - point1[0]) / (point1[0] - point_center[0])):
            return 0,minLen
        else:
            return max_speed,minLen
    return 0, minLen


# def side_of_speedometer
def get_Speedometer(max_speed):
    distance = 0
    radius = 0
    point_center = 0
    point1 = 0
    point2 = 0

    arcLength_values = []
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # blackened_image = threshold_image("green", frame)

    # TEST1-> determine types of green
    frame = cv2.imread('speedometer3.jpg')
    frame = imutils.resize(frame, width=600)
    # blackened_image = threshold_image("green", image_src)

    circles = getdot(frame)
    true_circles = circles

    if not circles:
        pass
    else:
        circles_copy = list(circles)

        center_points = circle_center(circles_copy)
        # circles_copy = copy.deepcopy(circles)
        i = 0

        lower_circle = find_zero_circle(center_points)
        center_points.remove(lower_circle)
        center_points.append(lower_circle)
        index_of_center = find_triangle(center_points, 2)
        right_circle = index_of_center
        center_points.remove(right_circle)
        upper_circle = center_points[0]

        new_flag, distance, radius, point1, point_center, point2 = circle_radius(lower_circle, upper_circle,
                                                                                 right_circle)
        if new_flag == False:
            return 0

        if point1[1] > point2[1]:
            zero_point = point1
            max_point = point2
        else:
            zero_point = point2
            max_point = point1

        base_angle = find_angle(point_center, point1)

        # This line is likely superfluous, as we do not ever need to know the linear distance between our two endpoints
        # Only arc length and proportions

        dist_between = dist_formula(point1, point2)

        # Why have I chose the sqare root of 2: I believ it has to do with the hypotenuse between two diagonal pixels of 1 pixel length
        # Note I have changed my initial algoritihm to be based around arc_Lengtha

        # just remember you wont have a perfect circle between points, but you can approximate

        arcLength_values, radius = circular_approx(radius, 1, base_angle, point2, dist_between, distance)

        # speed_arrow, slope = findArrow(frame, point_center)
        speed_arrow, slope = findArrow(frame, point_center)

        x_value = (point_center[0])
        y_value = (point_center)[1]
        intersection_point = (0, 0)

        # Remember the arlength_values are relative to the "origin" which is the value of X-value and y-Value
        intersection_point = find_intersection(x_value, y_value, arcLength_values, slope)

        if (intersection_point != (-1, -1)):
            true_speed = percentile_difference(point2, intersection_point, radius) * max_speed
            return true_speed
        if intersection_point == (-1, -1):
            if slope > ((point_center[1] - zero_point[0]) / (zero_point[0] - point_center[0])):
                return 0
            else:
                return max_speed
    return 0


def find_zero_circle(circles_point):
    max_x = - 2000
    max_y = -2000
    index = 0
    iterate = 0
    for coords in circles_point:
        if (coords[1] > max_y):
            max_y = coords[1]
            index = iterate
        iterate = iterate + 1
    return circles_point[index]


# Purpose to unpack the radius of the triangle of dots
# the edgetriangle will be what those rdii are
# unimportant_index should always have been the last one appended, in this case index 2
def find_triangle(circles_point, unimportant_index):
    triangle = []
    edges_triangle = []
    if len(circles_point) >= 3:
        triangle.append(dist_formula(circles_point[0], circles_point[1]))
        triangle.append(dist_formula(circles_point[0], circles_point[2]))
        triangle.append(dist_formula(circles_point[1], circles_point[2]))

        edges_triangle.append([triangle[0], triangle[1]])
        edges_triangle.append([triangle[0], triangle[2]])
        edges_triangle.append([triangle[1], triangle[2]])

        choice = predict_radius(edges_triangle, .15, 2)
        if choice == 0:
            return True, circles_point[0]
        elif choice == 1:
            return True, circles_point[1]
        # in this case the value has to be the right most point
        else:
            if circles_point[0][0] >= circles_point[1][0]:
                return True, circles_point[0]
            else:
                return True, circles_point[1]
    else:
        return False, 0


# use the edge traingel coordiantes to predict the least distance between points, if the difference is with in a certain error bound return a flag and a packed value
# Parameters: Edge TRaingles from find_triangels, actually just lengths of radii im comparing, sigma is errror bound, unimportant is the zero point

def predict_radius(edge_triangles, sigma, unimportant_index):
    edge_triangles.pop(unimportant_index)
    max_radius = 10000
    max_y = 10000
    index = 0
    iterate = 0
    value1 = abs(edge_triangles[0][0] - edge_triangles[0][1])
    avg_1 = (edge_triangles[0][0] + edge_triangles[0][1]) / 2
    value2 = abs(edge_triangles[1][0] - edge_triangles[1][1])
    avg_2 = (edge_triangles[1][0] + edge_triangles[1][1]) / 2

    if (value1) < (value2 + (avg_2 * sigma)):
        if (value1) < (value2 - (avg_2 * sigma)):
            return 0
        return 10

    else:
        return 1


# Look at PYImageSErach to edit and correct tthis code for my own purposes

def get_Speedometer_Digital(cam):
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    # cap = cv2.VideoCapture(0)

    # load the example image
    s, image = cam.read()

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break

    # extract the thermostat display, apply a perspective transform
    # to it
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if w >= 15 and (h >= 30 and h <= 40):
            digitCnts.append(c)
    # cv2.imshow("Input", image)
    # cv2.waitKey(0)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []

    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)

        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.5:
                on[i] = 1

                # lookup the digit and draw it on the image
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # display the digits
    print(u"{}{}.{} \u00b0C".format(*digits))
    # cv2.imshow("Input", image)
    # cv2.imshow("Output", output)
    test_flag = True
    # cv2.waitKey(0)


def get_Digital_Test():
    cap = cv2.VideoCapture(0)
    while (test_flag == False):
        get_Speedometer_Digital(cap)
        sleep(1)


cam1 = cv2.VideoCapture(0)
ret, frame = cam1.read()
        # frame = cv2.imread('MY_SPEEDOMETER3.jpg')
cv2.imshow("Frame", frame)
cv2.waitKey(0)
frame = imutils.resize(frame, width=600)
(height, width) = frame.shape[:2]
(flag, zero_point, max_point, arcLength_values, point_center, radius, radius_green_dot) = Speedometer_Calibration(80, frame)
if flag == False:
    time.sleep(20)
        # if flag == False:
        #    print("IT BROKE")
        #    time.sleep(10)
#point1, point2, height, frame, point_center, max_speed, radius
minLen = 95
speed, minLen = Speedometer_get_Speed(zero_point, max_point, height, frame, point_center, 80, radius, radius_green_dot, minLen)
print(speed)
sleep(60)