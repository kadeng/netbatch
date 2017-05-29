
import nnpy
import message_pb2 as msg
import time

sub = nnpy.Socket(nnpy.AF_SP, nnpy.SUB)
sub.connect('tcp://127.0.0.1:1235')
sub.setsockopt(nnpy.SUB, nnpy.SUB_SUBSCRIBE, '')

req = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
req.connect('tcp://127.0.0.1:1234')


br = msg.BatchRequest()
br.batch_id = 1
start = time.time()
req1 = br.record_requests.add()
req1.record_type = 1 # Recordfile record
req1.record_source_path="netbatch"
req1.record_source_indices.extend(reversed([2,5,6,19,23,34,56]))
req1.record_indices.extend(reversed(list(range(len(req1.record_source_indices)))))
#br.record_requests.extend([req1])
req.send(br.SerializeToString())
response = req.recv()
#print("Received response %d bytes - %d" % (len(response), response[0]))
expected = set(range(len(req1.record_source_indices)))

while(True):
    recd = sub.recv()
    rec = msg.Record()
    rec.ParseFromString(recd)
    expected.remove(rec.record_index)
    print("#%d - %d" % (rec.record_index, len(rec.data)))
#    print("Received %d bytes of data" % (len(rec.data)))
    if (len(expected)==0):
#        print("Received all records")
        break
stop = time.time()
print("Took %f ms" % (1000.0*(stop-start)))