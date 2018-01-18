try:
	import accimage as acc
except:
	acc = None
import random
import numpy as np
import cv2

class RandomRotate(object):

    def __init__(self, min_degree, max_degree):
        self.min_degree = min_degree
        self.max_degree = max_degree

    def __call__(self, img):
        deg = random.randint(self.min_degree*100, self.max_degree*100) / 100.0
        return img.rotate(deg, expand=False)


class AccimageToNumpy(object):

    def __call__(self, pic):

        if acc is not None and isinstance(pic, acc.Image):
            nppic = np.zeros([pic.height, pic.width, pic.channels], dtype=np.ubyte)
            pic.copyinterleaved(nppic)
            return nppic
        else:
            raise Exception("Unssupported source, has to be accimage image object")

def pil_to_numpy(img):
    ret = np.asarray(img)
    #print(ret.shape)
    return ret


def numpy_to_hsv(img_tensor):
    bufferhsv = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2HSV)
    return bufferhsv

def numpy_hwc_to_chw(img_tensor):
    try:
        return img_tensor.transpose([2,0,1])
    except Exception as e:
        #print(e)
        #print(img_tensor.shape)
        raise e

def byte_to_float(img_tensor):
    return img_tensor.astype(np.float32)/255.0
