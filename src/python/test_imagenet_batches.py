
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
sub = nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
#sub.connect('tcp://144.76.34.134:9877')
#sub.bind('tcp://127.0.0.1:9878')
sub.bind('ipc:///tmp/imgpipe.sock')

#sub.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, b'')
sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVBUF, 1024*1024*300)
sub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
sub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)

req = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
req.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
#req.connect('tcp://144.76.34.134:9876')
req.connect('tcp://127.0.0.1:9876')
br_init  = msg.BatchRequest()
br_init.batch_id = 0
lr = br_init.listing_requests.add()
lr.path=""
lr.file_extension=".idx"
lr.list_files=True
lr.list_dirs=False
lr.recurse=False
req.send(br_init.SerializeToString())
init_resp = req.recv()
print("INitial response length %d" % (len(init_resp)))
lresp = msg.BatchResponse()
lresp.ParseFromString(init_resp)
basenames = []
counts = []
for f in lresp.listing_response[0].files:
    basenames.append(f.path[:-4])
    counts.append(f.size/8)
recordcounts = dict(zip(basenames, counts))

print("Listed "+str(len(recordcounts)) + " record files")

br = msg.BatchRequest()
br.batch_id = 1
req1 = br.record_requests.add()
req1.record_type = 1 # Recordfile record
n_per_class = 20
n_total = 0
for i,bn in enumerate(basenames):
    req1.record_source_path=bn
    req1.record_source_indices.extend(sorted(list([int(val) for val in npr.randint(0,1299, n_per_class, np.int32)])))
    req1.record_indices.extend(list(range(i*n_per_class,((i+1)*n_per_class))))

#br.record_requests.extend([req1])
req.send(br.SerializeToString())
response = req.recv()
print("Received response %d bytes - %d" % (len(response), response[0]))

start = time.time()
expected = set(range(0, len(basenames)*n_per_class))
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
    try:
        expected.remove(rec.record_index)
    except:
        print("Unexpected record %d" % (rec.record_index))
    if (rec.error_code==msg.OK):
        #image = accimage.Image(bytes=rec.data)
        okcount+=1

    #print("#%d - %d - remaining %d" % (rec.record_index, len(rec.data), len(expected)))
#    print("Received %d bytes of data" % (len(rec.data)))
    if (len(expected)==0):
#        print("Received all records")
        break
stop = time.time()
print("Took %f ms - receive time %f ms, images/sec: %f, MB/sec: %f - ok: %d" % (1000.0*(stop-start), 1000.0*(stop-startrec), i/(stop-startrec), nbytes/(1024*1024*(stop-startrec)), okcount))