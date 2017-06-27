
from netbatch_dataset_bak import NetbatchImageDataset
import numpy.random as npr
import numpy as np
import cv2
import time
import math

nbi = NetbatchImageDataset(start_batch_id=int(time.time()*1000))
nbi.connect()
indexfiles = nbi.query_files("train-recs/", ".idx")
print(indexfiles)
for i, path in enumerate(sorted(indexfiles.keys())):
    rcount = int(indexfiles[path]/8)
    nbi.register_recordfile(path[:-4], rcount, 1.0, i)

print("Number of classes: %d" % (len(nbi.paths)))
nbi.set_batchsize(200)
nbi.start_sub()
nbi.request_batch(2)
import time
start = time.time()
totalbytes = 0
totalframes = 0
laststart = time.time()
for i, b in enumerate(nbi):
    batch, targets = b
    #nbi.request_batch(balance=True) # Never let the queue run dry
    frames = batch.size()[0]
    bytes = batch.size()[0]*batch.size()[1]*batch.size()[2]*batch.size()[3]*4
    totalframes += frames
    totalbytes+=bytes
    stop = time.time()
    t = stop-laststart
    total_t = stop-start
    idx = npr.randint(0,batch.size()[0], 5)
    print("Currently at %f frames/sec ; %f MB/sec, average %f frames/sec; %f MB/sec - Ignored %d"
          % ((frames/t), (bytes/(t*1024*1024)), totalframes/total_t, (totalbytes/(total_t*1024*1024)), nbi.ignored_count))
    laststart = time.time()
