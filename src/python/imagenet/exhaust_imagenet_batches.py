
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
#import accimage
sub = nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
#sub.connect('tcp://144.76.34.134:9877')
#sub.bind('tcp://127.0.0.1:9878')
sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVBUF, 1024*1024*300)
sub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
sub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
sub.bind('ipc:///tmp/imgpipe.sock')

#sub.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, b'')


start = time.time()
startrec = time.time()
i = 0
nbytes = 0
okcount = 0
while(True):
    recd = sub.recv()
    #print("Recv %d bytes" % (len(recd)))
    nbytes += len(recd)
    rec = msg.Record()
    rec.ParseFromString(recd)
    i+=1
    if (rec.error_code==msg.OK):
        #image = accimage.Image(bytes=rec.data)
        okcount+=1
    if (i % 100==1):
        print("Received %d" % (i))
    #print("#%d - %d - remaining %d" % (rec.record_index, len(rec.data), len(expected)))
#    print("Received %d bytes of data" % (len(rec.data)))
