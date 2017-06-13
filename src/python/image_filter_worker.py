
import nnpy
import message_pb2 as msg
import time
import os
import glob
import json
import numpy.random as npr
import numpy as np
import os.path
import cv2
import accimage
import torchvision.transforms as transforms
import numpy
sub = nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVBUF, 1024*1024*300)
sub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
sub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)

#sub.connect('tcp://144.76.34.134:9877')
sub.connect('tcp://127.0.0.1:9877')
pub = nnpy.Socket(nnpy.AF_SP, nnpy.PUSH)
pub.setsockopt(nnpy.SOL_SOCKET, nnpy.SNDBUF, 1024*1024*300)
pub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
#pub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
#pub.connect('tcp://127.0.0.1:9878')
pub.connect('ipc:///tmp/imgpipe.sock')

def tensor_to_numpy_hsv(img_tensor):
    buffer = img_tensor.numpy()
    b2 = buffer.transpose([1,2,0])
    bufferhsv = cv2.cvtColor(b2, cv2.COLOR_RGB2HSV)
    return bufferhsv

def print_img_size(img):
    if (img is not None):
        print(img.size)
    return img

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
img_transforms = transforms.Compose([
            transforms.Scale(264),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#            normalize,
            tensor_to_numpy_hsv
        ])

start = time.time()
startrec = time.time()
i = 0
nbytes = 0
okcount = 0
print("Working ...")
while(True):
    recd = sub.recv()
    #print("Recv %d bytes" % (len(recd)))
    nbytes += len(recd)
    rec = msg.Record()
    rec.ParseFromString(recd)
    i+=1
    if (rec.error_code==msg.OK):
        try:
            image = accimage.Image(bytes=rec.data)
            img_ndarr = img_transforms(image)
            #img = (img_ndarr*255.0).astype(np.uint8)
            #img_bgr=cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            #cv2.imwrite("/tmp/img%d.jpg" % (rec.record_index), img_bgr)
            rec.route_id=0
            rec.data = img_ndarr.tobytes()
            pub.send(rec.SerializeToString())
            okcount+=1
        except:
            rec.route_id = 0
            rec.data = b""
            rec.error_code=msg.FILE_FORMAT_ERROR
            pub.send(rec.SerializeToString())
    else:
        print("ERROR") #pub.send(recd)
        print(rec)
