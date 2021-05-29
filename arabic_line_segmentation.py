import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from findpeaks import findpeaks



def erode(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    res = cv.erode(img, kernel)
    return res


def dilate(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    res = cv.dilate(img, kernel)
    return res


def findLocalMins(arr, threshold):
    currentMinimumIdx = 0
    lastWasPeak = False
    out = []
    for i in range(len(arr)):
        if arr[i] > threshold and not lastWasPeak:
            out.append(currentMinimumIdx)
            lastWasPeak = True
            currentMinimumIdx = i
        elif arr[i] < threshold:
            lastWasPeak = False
            if arr[i] < arr[currentMinimumIdx]:
                currentMinimumIdx = i
    return out



if __name__ == '__main__':
    img_src = "OCR samples/1.jpeg"
    img = cv.imread(img_src)
    img = cv.resize(img, (800, 800))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # TODO: Add a filter to remove noise
    img = cv.bitwise_not(img)
    (thresh, img) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    cv.imshow("not", img)
    dilated = dilate(img)
    cv.imshow("dilation", dilated)
    eroded = erode(dilated)
    cv.imshow("erosion", eroded)
    row_sum = np.sum(img, axis=1)
    col_sum = np.sum(img, axis=0)
    average = np.average(row_sum, axis=0)
    plt.plot(row_sum.tolist())
    plt.show()
    fp = findpeaks(lookahead=10)
    peaks = fp.fit(row_sum)
    fp.plot()
    valleys = []
    for index, row in peaks['df'].iterrows():
        if row['valley'] :
            valleys.append(index)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for line in valleys:
        img = cv.line(img, (0, line), (len(img[0]), line), (0,255,0),thickness=1)
    cv.imshow("Lines", img)
    cv.waitKey()
