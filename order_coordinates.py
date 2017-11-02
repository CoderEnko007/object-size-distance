# -*- coding:utf-8 -*-
from scipy.spatial import distance as dist
from imutils import contours
import numpy as np
import imutils
import argparse
import cv2

def order_points(pts):
    Sorts = pts[np.argsort(pts[:, 0]), :]
    leftMost = Sorts[:2, :]
    rightMost = Sorts[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 0]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype=np.float)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

boundingBoxes = np.array([cv2.boundingRect(c) for c in cnts])
# left to right
cnts = np.array(cnts)[np.argsort(boundingBoxes[:, 0])]
# reverse
# cnts = np.array(cnts)[np.argsort(boundingBoxes[:, 0])[::-1]]
# #cnts = sorted(zip(cnts, boundingBoxes), key=lambda t: t[1][0], reverse=False)
# (cnts, _) = contours.sort_contours(cnts)
# boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#print(zip(cnts, boundingBoxes)[1][1])
# lambda为匿名函数，b[1][0]表示zip（这里是tuple）中的第二个元素(即boundingBoxes)的第0个元素(boundingBox的左上角x1坐标)
# 一个星号表示参数为一个元祖，那么输出也就是一组元祖
# (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                     key=lambda b: b[1][0], reverse=False))
# a = zip(sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][0], reverse=False))
# print(a[0][0])
# print(type(a[0][0]))
# print(boundingBoxes)
# print(a[])
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

for (i, c) in enumerate(cnts):
    if cv2.contourArea(c) < 100:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box) if imutils.is_cv3() else cv2.cv.BoxPoints(box)
    box = np.array(box, dtype=np.int)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    print("Object #%s" % str(i + 1))
    cv2.imshow("image", image)

    rect = order_points(box)
    print(rect.astype("int"))
    print("")

    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
    cv2.putText(image, "Object #{}".format(i + 1),
                (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
