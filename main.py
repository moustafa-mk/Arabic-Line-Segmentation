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
    eroded = erode(img)
    cv.imshow("erosion", eroded)
    dilated = dilate(eroded)
    cv.imshow("dilation", dilated)
    cv.waitKey()
