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


def smooth(img):
    res = cv.medianBlur(img, 3)
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
    orig_img = cv.imread(img_src)
    orig_img = cv.resize(orig_img, (800, 800))
    img = smooth(orig_img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bitwise_not(img)
    (thresh, img) = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    dilated = dilate(img)
    eroded = erode(dilated)
    row_sum = np.sum(img, axis=1)
    col_sum = np.sum(img, axis=0)
    average = np.average(row_sum, axis=0)
    plt.plot(row_sum.tolist())
    plt.show()
    fp = findpeaks(lookahead=10)
    peaks = fp.fit(row_sum)
    fp.plot()
    lastTroughIdx = 0
    valleys = []

    for index, row in peaks['df'].iterrows():
        if row['valley']:
            values, bins, _ = plt.hist(row_sum[lastTroughIdx: index])
            area = sum(np.diff(bins) * values)
            print("Peak size: %s,%s -> %s; Area: %s" % (str(index), str(lastTroughIdx),str(index - lastTroughIdx), str(area)))
            lastTroughIdx = index
            valleys.append(index)
    # orig_img = cv.cvtColor(orig_img, cv.COLOR_GRAY2RGB)
    for line in valleys:
        orig_img = cv.line(orig_img, (0, line), (len(orig_img[0]), line), (0, 255, 0), thickness=1)
    cv.imshow("Lines", orig_img)
    cv.waitKey()
