
import nnpy
import message_pb2 as msg
import time
import matplotlib.pylab as plt
import numpy as np
import cv2
import accimage
import torchvision.transforms as transforms
from image_transforms import AccimageToNumpy, numpy_to_hsv,RandomRotate, pil_to_numpy

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

img_transforms = transforms.Compose([
            AccimageToNumpy(),
            transforms.ToPILImage(),
            RandomRotate(-5, 5),
            transforms.Scale(244),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            pil_to_numpy,
            numpy_to_hsv
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
            rec.route_id=0
            rec.data = img_ndarr.tobytes()
            pub.send(rec.SerializeToString())
            okcount+=1
        except:
            import traceback
            traceback.print_exc()
            rec.route_id = 0
            rec.data = b""
            rec.error_code=msg.FILE_FORMAT_ERROR
            pub.send(rec.SerializeToString())
    else:
        print("ERROR") #pub.send(recd)
        print(rec)
