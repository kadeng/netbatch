
from netbatch_dataset import NetbatchImageDataset
import numpy.random as npr

nbi = NetbatchImageDataset()
nbi.connect()
indexfiles = nbi.query_files("train-recs/", ".idx")
print(indexfiles)
for i, path in enumerate(indexfiles.keys()):
    rcount = int(indexfiles[path]/8)
    nbi.register_recordfile(path[:-4], rcount, 1.0)

nbi.set_batchsize(1000)
nbi.start_sub()
nbi.request_batch(1000, 5)
import time
start = time.time()
totalbytes = 0
totalframes = 0
for i in range(100):
    laststart = time.time()
    batch = nbi.next_batch()
    frames = batch.shape[0]
    bytes = batch.shape[0]*batch.shape[1]*batch.shape[2]*batch.shape[3]*4
    totalframes += frames
    totalbytes+=bytes
    stop = time.time()
    t = stop-laststart
    total_t = stop-start
    print("Currently at %f frames/sec ; %f MB/sec, average %f frames/sec; %f MB/sec"
          % ((frames/t), (bytes/(t*1024*1024)), totalframes/total_t, (totalbytes/(total_t*1024*1024))))
