import torch.utils.data as data
import threading
from queue import Queue
from collections import defaultdict
import nnpy
import message_pb2 as msg
import numpy as np
import numpy.random as npr
import torch

class NetbatchImageDataset(data.Dataset):

    def __init__(self, source, record_shape=(3, 224,224), record_dtype=np.float32, target_dtype=np.int64):
        self.source = source
        self.record_dtype = record_dtype
        self.target_dtype = target_dtype
        self.batch_queue = Queue()
        self.paths = list()
        self.path_types = list()
        self.weights = list()
        self.recordcounts = list()
        self.normweights = None
        self.record_shape = list(record_shape)
        self.record_pos = list()
        self.shuffle_indices = list()
        self.batchsize = None
        self.expecting_batches = set()
        self.targets = list()
        self.batch_targets = dict()
        self.batch_recordcounts = defaultdict(lambda: 0)

    def accept(self, rec):
        if (rec.batch_id not in self.expecting_batches):
            return False
        if (rec.error_code == msg.OK and rec.data is not None):
            b1 = np.frombuffer(rec.data, dtype=self.record_dtype)
            recarr = np.reshape(b1, newshape=self.record_shape)
            self.batch_recordcounts[rec.batch_id] += 1
            self.batch_arrays[rec.batch_id][rec.record_index, :, :, :] = recarr
            self.batch_indices[rec.batch_id].add(rec.record_index)
        else:
            print("Error %r" % (rec.error_code))
            self.batch_recordcounts[rec.batch_id] += 1
            self.batch_arrays[rec.batch_id][rec.record_index, :, :, :] = 0.0
            self.batch_indices[rec.batch_id].add(rec.record_index)
        if (self.batch_recordcounts[rec.batch_id] >= self.batchsize):
            # print("Recordcount %d for batch %d reached - set size %d" % (batch_recordcounts[rec.batch_id],
            #                                                             rec.batch_id, len(batch_indices[rec.batch_id])))
            self.batch_queue.put(
                (self.batch_arrays[rec.batch_id], np.array(self.batch_targets[rec.batch_id], dtype=self.target_dtype)))
            del self.batch_targets[rec.batch_id]
            del self.batch_arrays[rec.batch_id]
            del self.batch_recordcounts[rec.batch_id]
            del self.batch_indices[rec.batch_id]
            self.expecting_batches.remove(rec.batch_id)
        return True

    def set_batchsize(self, batchsize):

        if (self.batchsize==batchsize and self.norm_weights is not None):
            return

        self.batchsize = batchsize
        self.batch_shape = tuple([self.batchsize] + self.record_shape)
        self.batch_arrays = defaultdict(lambda: np.zeros(self.batch_shape, dtype=self.record_dtype))
        self.batch_indices = defaultdict(lambda: set())

        self.mincount = np.zeros((len(self.weights)), dtype=np.int32)
        self.fixed_count = 0
        weights = np.array(self.weights, dtype=np.float32)
        normweights = weights / np.sum(weights)
        self.norm_weights = normweights
        expected = normweights * float(batchsize)
        mincount = np.floor(expected)
        remaining = expected - mincount
        sremain = np.sum(remaining)
        if (sremain>0.0):
            normremaining = remaining / sremain
            self.remaining_normweights = normremaining
        else:
            self.remaining_normweights = None
        self.balanced_minimum_counts = mincount.astype(np.int32)
        self.fixed_count = np.sum(self.balanced_minimum_counts)

    def register_recordfile(self, path, recordcount, weight, target=None):
        self.paths.append(path)
        self.path_types.append(type)
        self.targets.append(target)
        self.recordcounts.append(recordcount)
        self.weights.append(weight)
        self.record_pos.append(0)
        si = np.arange(start=0, stop=recordcount, dtype=np.int32)
        npr.shuffle(si)
        self.shuffle_indices.append(si)
        self.normweights=None

    def request_batch(self, nbatches=1, balance=True):
        batchsize = self.batchsize

        if (balance):
            # We enforce a balanced dataset
            if (self.remaining_normweights is not None and self.fixed_count<batchsize):
                pcounts = npr.multinomial(batchsize-self.fixed_count, self.remaining_normweights, nbatches)
                for i in range(len(pcounts)):
                    pcounts[i] = np.array(pcounts[i])+self.balanced_minimum_counts
            else:
                pcounts = [ self.balanced_minimum_counts ] * nbatches
        else:
            pcounts = npr.multinomial(batchsize, self.norm_weights, nbatches)

        for b in range(nbatches):
            bpcounts = pcounts[b]


            br = msg.BatchRequest()
            self.source.batch_id += 1
            br.batch_id = self.source.batch_id

            ridx = np.arange(0, batchsize, dtype=np.int32)
            npr.shuffle(ridx)
            target_record_indices = [int(t) for t in ridx]
            pos = 0
            bt = [None]*batchsize
            self.batch_targets[br.batch_id] = bt
            for i, path in enumerate(self.paths):
                target = self.targets[i]
                pcount = bpcounts[i]
                if (pcount<=0):
                    continue
                rpos = self.record_pos[i]
                sindices = self.shuffle_indices[i]
                if (rpos + pcount>sindices.shape[0]):
                    npr.shuffle(sindices)
                    self.record_pos[i] = 0
                    rpos = 0
                si = sindices[rpos:(rpos+pcount)]
                self.record_pos[i]+=pcount
                req1 = br.record_requests.add()
                req1.record_type = 1  # Recordfile record
                req1.record_source_path = path
                src_indices = sorted([int(j) for j in si])
                trg_indices = target_record_indices[pos:(pos+pcount)]

                if (callable(target)):
                    for sid, tid in zip(src_indices, trg_indices):
                        bt[tid] = target(sid)
                elif (isinstance(target, dict)):
                    for sid, tid in zip(src_indices, trg_indices):
                        bt[tid] = target[sid]
                else:
                    for sid, tid in zip(src_indices, trg_indices):
                        bt[tid] = target
                req1.record_source_indices.extend(src_indices)
                req1.record_indices.extend(trg_indices)
                pos += pcount
            self.expecting_batches.add(br.batch_id)
            response = self.source.send_batch_request(br)

    def next_batch(self, as_torch=True):
        ret, target = self.batch_queue.get()
        if (as_torch):
            ret = torch.from_numpy(ret)
            target = torch.from_numpy(target)
        return ret, target

    def __next__(self):
        ret = self.next_batch()
        self.request_batch()
        return ret

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self


class NetBatchSource(object):

    def __init__(self, start_batch_id=1, sub_url='ipc:///tmp/imgpipe.sock', req_url='tcp://127.0.0.1:9876'):
        self.batch_id = start_batch_id
        self.sub_url = sub_url
        self.req_url = req_url
        self.receivers = list()

    def add_receiver(self, receiver):
        self.receivers.append(receiver)

    def connect(self):
        self.req = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
        self.req.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
        self.req.connect(self.req_url)
        self.ignored_count = 0

    def query_files(self, basepath, file_extension):
        br_init = msg.BatchRequest()
        br_init.batch_id = 0
        lr = br_init.listing_requests.add()
        lr.path = basepath
        lr.file_extension = file_extension
        lr.list_files = True
        lr.list_dirs = False
        lr.recurse = False
        self.req.send(br_init.SerializeToString())
        init_resp = self.req.recv()
        #print("INitial response length %d" % (len(init_resp)))
        lresp = msg.BatchResponse()
        lresp.ParseFromString(init_resp)
        basenames = []
        counts = []
        for f in lresp.listing_response[0].files:
            basenames.append(f.path)
            counts.append(f.size)
        recordcounts = dict(zip(basenames, counts))
        return recordcounts

    def start_sub(self):
        sub = nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
        sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVBUF, 1024 * 1024 * 600)
        sub.setsockopt(nnpy.SOL_SOCKET, 16, 1024 * 1024 * 300)
        sub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
        sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVTIMEO, 1000*10)
        sub.bind('ipc:///tmp/imgpipe.sock')
        self.sub = sub


        def receive_batches():
            try:
                while(True):
                    while (True):
                        try:
                            rrec = self.sub.recv()
                            break
                        except nnpy.NNError as e:
                            if (e.error_no==nnpy.ETIMEDOUT):
                                for rec in self.receivers:
                                    if (len(rec.expecting_batches)>0):
                                        print("Timeout, receiving batches - despite expecting %d - requesting missing batches" % (len(rec.expecting_batches)))
                                        rec.request_batch(len(rec.expecting_batches))
                                continue
                            raise e
                    rec = msg.Record()
                    rec.ParseFromString(rrec)
                    accepted = False
                    for receiver in self.receivers:
                        if (receiver.accept(rec)):
                            accepted = True
                            break
                    if (not accepted):
                        self.ignored_count += 1

            finally:
                print("Finished receiving batches")

        thr = threading.Thread(target=receive_batches)
        thr.daemon = True
        thr.start()

    def send_batch_request(self, br):
        self.req.send(br.SerializeToString())
        response = self.req.recv()
        res = msg.BatchResponse()
        res.ParseFromString(response)
