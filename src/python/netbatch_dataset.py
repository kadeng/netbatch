import torch.utils.data as data
import threading
from queue import Queue
from collections import defaultdict
import nnpy
import message_pb2 as msg
import numpy as np
import numpy.random as npr

class NetbatchImageDataset(data.Dataset):

    def __init__(self, start_batch_id=1, sub_url='ipc:///tmp/imgpipe.sock', req_url='tcp://127.0.0.1:9876',
                 record_shape=(224,224,3)):
        self.sub_url = sub_url
        self.req_url = req_url
        self.batch_queue = Queue()
        self.paths = list()
        self.weights = list()
        self.recordcounts = list()
        self.batch_id = start_batch_id
        self.normweights = None
        self.record_shape = list(record_shape)
        self.record_pos = list()
        self.shuffle_indices = list()
        self.batchsize = None

    def connect(self):
        self.req = nnpy.Socket(nnpy.AF_SP, nnpy.REQ)
        self.req.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
        # req.connect('tcp://144.76.34.134:9876')
        self.req.connect(self.req_url)

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

        sub.bind('ipc:///tmp/imgpipe.sock')

        self.sub = sub
        batch_recordcounts = defaultdict(lambda: 0)
        batch_shape = tuple([self.batchsize] + self.record_shape)
        batch_arrays = defaultdict(lambda: np.zeros(batch_shape, dtype=np.ubyte))
        batch_indices = defaultdict(lambda: set())

        def receive_batches():
            print("Receiving batches")
            while(True):
                rrec = self.sub.recv()
                rec = msg.Record()
                rec.ParseFromString(rrec)
                if (rec.error_code==msg.OK and rec.data is not None):
                    b1 = np.frombuffer(rec.data, dtype=np.ubyte)
                    recarr = np.reshape(b1, newshape=self.record_shape)
                    batch_recordcounts[rec.batch_id] += 1
                    batch_arrays[rec.batch_id][rec.record_index,:,:,:]=recarr
                    batch_indices[rec.batch_id].add(rec.record_index)
                else:
                    print("Error %r" % (rec.error_code))
                    batch_recordcounts[rec.batch_id] += 1
                    batch_arrays[rec.batch_id][rec.record_index, :, :, :] = 0.0
                    batch_indices[rec.batch_id].add(rec.record_index)
                if (batch_recordcounts[rec.batch_id] >= self.batchsize):
                    print("Recordcount %d for batch %d reached - set size %d" % (batch_recordcounts[rec.batch_id],
                                                                                 rec.batch_id, len(batch_indices[rec.batch_id])))
                    self.batch_queue.put(batch_arrays[rec.batch_id])
                    del batch_arrays[rec.batch_id]
                    del batch_recordcounts[rec.batch_id]
                    del batch_indices[rec.batch_id]

        thr = threading.Thread(target=receive_batches)
        thr.daemon = True
        thr.start()

    def set_batchsize(self, batchsize):
        if (self.batchsize==batchsize and self.norm_weights is not None):
            return
        self.batchsize = batchsize
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

    def register_recordfile(self, path, recordcount, weight):
        self.paths.append(path)
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
            if (self.remaining_normweights is not None):
                pcounts = npr.multinomial(batchsize, self.remaining_normweights, nbatches)
                for i in range(len(pcounts)):
                    pcounts[i] = np.array(pcounts[i])+self.balanced_minimum_counts
            else:
                pcounts = [ self.balanced_minimum_counts ] * nbatches
        else:
            pcounts = npr.multinomial(batchsize, self.norm_weights, nbatches)

        for b in range(nbatches):
            bpcounts = pcounts[b]
            self.batch_id += 1
            br = msg.BatchRequest()
            br.batch_id = self.batch_id
            ridx = np.arange(0, batchsize, dtype=np.int32)
            npr.shuffle(ridx)
            target_record_indices = [int(t) for t in ridx]
            pos = 0
            for i, path in enumerate(self.paths):
                pcount = bpcounts[i]
                if (pcount<=0):
                    continue
                rpos = self.record_pos[i]
                sindices = self.shuffle_indices[i]
                if (rpos + pcount>sindices.shape[0]):
                    sindices = npr.shuffe(sindices)
                    self.shuffle_indices[i] = sindices
                    self.record_pos[i] = pcount
                si = sindices[rpos:(rpos+pcount)]
                self.record_pos[i]+=pcount
                req1 = br.record_requests.add()
                req1.record_type = 1  # Recordfile record
                req1.record_source_path = path
                req1.record_source_indices.extend(sorted([int(j) for j in si]))
                req1.record_indices.extend(target_record_indices[pos:(pos+pcount)])
                pos += pcount
            self.req.send(br.SerializeToString())
            response = self.req.recv()

    def next_batch(self):
        return self.batch_queue.get()

        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
