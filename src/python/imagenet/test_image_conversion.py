import accimage as acc
import glob
import random

import cv2
import torchvision.transforms as transforms

from imagenet.image_transforms import AccimageToNumpy, numpy_to_hsv, RandomRotate, pil_to_numpy

images = glob.glob("/data/datasets/imagenet/imagenet-data/train/n02129604/*.JPEG")
random.shuffle(images)

#img = acc.Image('/data/datasets/imagenet/imagenet-data/train/n02129604/n02129604_7265.JPEG')

for i in range(100):
    img = acc.Image(images[i])

    tr8 = transforms.Compose([
        AccimageToNumpy(),
        transforms.ToPILImage(),
        RandomRotate(-5, 5),
        transforms.Scale(244),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        pil_to_numpy,
        numpy_to_hsv
    ])
    trg3hsv = tr8(img)
    trg3bgr = cv2.cvtColor(trg3hsv, cv2.COLOR_HSV2BGR)
    trg3rgb = cv2.cvtColor(trg3hsv, cv2.COLOR_HSV2RGB)

    cv2.imwrite("/tmp/test3.jpg", trg3bgr)
    import matplotlib.pylab as plt
    plt.title(images[i])
    plt.imshow(trg3rgb)
    plt.show()

print("DONE")