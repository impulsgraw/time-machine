import numpy as np
import cv2 as cv
import helper.common as paths


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        cv.imshow(self.windowname + ": Mask", self.dests[1])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None
        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


def init_image(img_path, options=None):
    options = {"smoothing": False, **(options if type(options) == dict else {})}

    img = cv.imread(img_path, cv.IMREAD_COLOR)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if options["smoothing"]:
        img = cv.bilateralFilter(img, 5, 75, 75)

    return img


def inpaint(image):
    img_mask = image.copy()
    inpaintMask = np.zeros(image.shape[:2], np.uint8)
    sketch = Sketcher("Image", [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))

    while True:
        a = cv.waitKey(0)
        if a == 27:
            break
        if a == ord('n'):
            result = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
            cv.imshow("Inpaint", result)
    return result


def scale(res, height, width):
    return cv.resize(res, (width, height), interpolation=cv.INTER_CUBIC)


def output(images, img_path):
    cv.imwrite(img_path, np.hstack(images))

def show(images):
    cv.imshow("Result", np.hstack(images))
    cv.waitKey()