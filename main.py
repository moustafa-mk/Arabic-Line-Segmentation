import cv2 as cv


def erode(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    res = cv.erode(img, kernel)
    return res


def dilate(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    res = cv.dilate(img, kernel)
    return res


if __name__ == '__main__':
    img_src = "OCR samples/sample2.png"
    img = cv.imread(img_src)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bitwise_not(img)
    cv.imshow("not", img)
    dilated = dilate(img)
    cv.imshow("dilation", dilated)
    eroded = erode(dilated)
    cv.imshow("erosion", eroded)
    cv.waitKey()
