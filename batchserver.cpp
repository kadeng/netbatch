//============================================================================
// Name        : batchserver.cpp
// Author      : Kai Londenberg
// Version     :
// Copyright   : Apache License V2.0
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <chrono>
#include <nanomsg/reqrep.h>
#include <nanomsg/pubsub.h>
#include <dlib/dir_nav.h>
#include "nn.hpp" // Nanomsg C++ interface
#include <nanomsg/tcp.h>
#include "tqueue.hpp"
#include "proto/message.pb.h"
#include "argparse.hpp" // See https://github.com/hbristow/argparse
#include <memory>
#include <bits/unique_ptr.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>

using namespace std;
using namespace netbatch;

struct record_data {
    char *start;
    char *end;

    record_data(char *start, char *end) : start(start),end(end) {}
};

class RecordfileReader {

    unsigned int rcount;
    streamsize* index;
    char *data_mmap;
    int data_fd;
    size_t data_size;
    bool is_open;

        static size_t filesize(const string &path) {
            struct stat st;
            stat(path.c_str(), &st);
            if (st.st_size>=0) {
                return (size_t) st.st_size;
            }
            return 0;
        }

    public:
        const string basepath;

        RecordfileReader(const string &basepath) : basepath(basepath), rcount(0),is_open(false) {
        }

        ~RecordfileReader() {
            close_recordfile();
        }



        unsigned int size() {
            return rcount;
        }

        record_data at(const unsigned int idx) {
            if (!is_open || idx>=rcount) {
                return record_data(nullptr, nullptr);
            }
            char *start = data_mmap;
            if (idx>0) {
                start = &start[index[idx-1]];
            }
            char *end = &data_mmap[index[idx]];
            return record_data(start, end);
        }

        bool open_recordfile(bool pre_populate) {
            if (is_open) {
                return true;
            }
            string recordfilename = basepath +".rec";
            string indexfilename = basepath +".idx";

            size_t fsize = filesize(recordfilename);
            size_t isize = filesize(indexfilename);

            if (fsize<=0 || isize<=0 || (isize % sizeof(streamsize)!=0)) {
                return false;
            }

            rcount = isize / sizeof(streamsize);

            // Read index entirely into memory
            int index_fd = open(indexfilename.c_str(), O_RDONLY, 0);
            void* midx = mmap(NULL, isize, PROT_READ, MAP_PRIVATE, index_fd, 0);
            if (midx == MAP_FAILED) {
                 close(index_fd);
                 return false;
            }
            this->index = (streamsize*)malloc(isize);
            memcpy(this->index, midx, isize);
            munmap(midx, isize);
            close(index_fd);

            // MMap data records
            data_fd = open(recordfilename.c_str(), O_RDONLY, 0);
            if (data_fd==-1) {
                free(this->index);
                return false;
            }

            int mmap_flags = MAP_SHARED; // Compare against MAP_PRIVATE, just in case there's a performance differe
            if (pre_populate) {
                mmap_flags |= MAP_POPULATE;
            }
            data_size = fsize;
            void* mmappedData = mmap(NULL, fsize, PROT_READ, mmap_flags, data_fd, 0);
            if (mmappedData == MAP_FAILED) {
                free(this->index);
                close(data_fd);
                return false;
            }
            data_mmap = (char*)mmappedData;
            is_open = true;
            return true;
        }

        void close_recordfile() {
            if (!is_open) {
                return;
            }
            is_open = false;
            free(this->index);
            rcount = 0;
            this->index = nullptr;
            munmap(data_mmap, data_size);
            close(data_fd);
            data_mmap = nullptr;
            data_fd = -1;
        }

};

class NNBuffer {
    const void *buffer;
    const size_t len;

    public:
        NNBuffer(void *buffer, size_t len) : buffer(buffer), len(len) {

        }

        ~NNBuffer() {
            if (buffer!=nullptr) {
                nn_freemsg(const_cast<void*>(buffer));
            }
        }

        const void *pointer() {
            return buffer;
        }

        const size_t length() {
            return len;
        }
};


struct NetBatchRequest {
    BatchRequest batch_request;
    uint32_t completed_requests = 0;

};

struct NetbatchServer {
    string server_url;
    string broadcast_url;
    string basepath;
    size_t max_record_size;
    mutex server_mutex;

    TQueue<const RecordsRequest*> fqueue;
    TQueue<shared_ptr<Record>> response_queue;
    map<uint32_t, shared_ptr<NetBatchRequest>> requested_batches;
    map<string,shared_ptr<RecordfileReader>> recordfiles;
    bool stop = false;

    void set_basepath(string basepath_) {
        if (basepath_.size()==0) {
            basepath = "/";
        } else if (basepath_[basepath_.size()-1]!='/') {
            basepath = basepath_ + "/";
        } else {
            basepath = basepath_;
        }
    }
};


class BatchRequestAcceptor {
        nn::socket *sock;
        NetbatchServer &netbatch;

	public:

        BatchRequestAcceptor(NetbatchServer &netbatch) :
            sock(nullptr),
            netbatch(netbatch)

            {
		}

        virtual ~BatchRequestAcceptor() {
		}


        void start() {
            sock = new nn::socket(AF_SP, NN_REP);
            int nodelay = 1;
            try {
                sock->setsockopt(NN_TCP, NN_TCP_NODELAY, (char*) &nodelay,
                                 sizeof (nodelay));
                int rcvbufsize = 1024*1024*8;
                sock->setsockopt(NN_SOL_SOCKET, NN_RCVBUF, (char*) &rcvbufsize,
                                 sizeof (rcvbufsize));
            } catch(nn::exception &e) {
                cerr << e.what() << endl;
                exit(-1);
            }

            sock->bind(netbatch.server_url.c_str());
            while(1) {
                void *buf = nullptr;
                int ret = sock->recv(&buf, NN_MSG, 0);
                if (ret>0) {
                    NNBuffer nnbuf(buf, ret);
                    auto netbatch_request = make_shared<NetBatchRequest>();
                    if (netbatch_request->batch_request.ParseFromArray(nnbuf.pointer(), nnbuf.length())) {

                        {
                            unique_lock<mutex> lock(netbatch.server_mutex);
                            if (netbatch.requested_batches[netbatch_request->batch_request.batch_id()]) {
                                lock.unlock();
                                // Duplicate request
                                cerr << "Duplicate request for batch id " << netbatch_request->batch_request.batch_id();
                                sock->send("\2", 1, 0);
                                continue;
                            } else {
                                BatchRequest &br = netbatch_request->batch_request;
                                netbatch.requested_batches[br.batch_id()] = netbatch_request;
                                lock.unlock();
                                for (int j=0;j<br.record_requests_size();j++) {
                                    RecordsRequest *rr = br.mutable_record_requests(j);
                                    rr->set_batch_id(br.batch_id());
                                    netbatch.fqueue.push(rr);
                                }
                                sock->send("\0", 1, 0);
                            }
                        }

                    } else {
                        sock->send("\1", 1, 0);
                    }
                }
            }
		}


};

class RecordRequestWorker {
        NetbatchServer &netbatch;

    public:

        RecordRequestWorker(NetbatchServer &netbatch) :
            netbatch(netbatch) {
        }

        ~RecordRequestWorker() {
        }

        void start() {
            while(1) {
                const RecordsRequest *rreq = netbatch.fqueue.pop();
                //cout << "Received records request in thread " << std::this_thread::get_id() << endl;
                handleRecordsRequest(rreq);
            }
        }

    private:

        void handleRecordsRequest(const RecordsRequest *rreq) {
            auto breq = netbatch.requested_batches[rreq->batch_id()];
            if (!breq) {
                throw std::runtime_error("Fatal error, no batch request for records request");
            }
            if (rreq->record_type()==RecordType::FILE) {
                auto rec = make_shared<Record>();
                rec->set_batch_id(rreq->batch_id());

                if (rreq->record_source_indices_size()>0) {
                    rec->set_record_index(rreq->record_source_indices(0));
                } else {
                    rec->set_record_index(0);
                }
                string::size_type spos = rreq->record_source_path().find("..");
                string::size_type ssize = rreq->record_source_path().size();
                if (ssize<500 && spos==string::npos) {
                    // Now, we set the data of the record to the file data, without copying it unneccessarily..

                        ifstream file(netbatch.basepath + rreq->record_source_path(), std::ios::binary | std::ios::ate);
                        if (file) {
                            streamsize size = file.tellg();
                            if (size>netbatch.max_record_size) {
                                rec->set_error_code(ErrorCode::TOO_LARGE);
                            } else {
                                file.seekg(0, std::ios::beg);
                                string *data = rec->mutable_data();
                                data->resize(size);
                                char *buf = const_cast<char*>(data->c_str()); // Yes, bad things might happen. But don't ..
                                file.read(buf, size);
                            }
                        } else {
                            rec->set_error_code(ErrorCode::FILE_OPEN_FAILED);
                        }

                } else {
                    rec->set_error_code(ErrorCode::INVALID_PATH);
                }
                netbatch.response_queue.push(std::move(rec));

            }  else if (rreq->record_type()==RecordType::RECORDFILE_RECORD) {
                // We return one record for each record index
                shared_ptr<RecordfileReader> recfile;
                string::size_type spos = rreq->record_source_path().find("..");
                string::size_type ssize = rreq->record_source_path().size();

                if (ssize<500 && spos==string::npos) {
                    unique_lock<mutex> lock_(netbatch.server_mutex);
                    auto riter = netbatch.recordfiles.find(rreq->record_source_path());
                    if (riter==netbatch.recordfiles.end()) {
                        recfile = make_shared<RecordfileReader>(netbatch.basepath + rreq->record_source_path());
                        netbatch.recordfiles[rreq->record_source_path()] = recfile;
                    } else {
                        recfile = netbatch.recordfiles[rreq->record_source_path()];
                    }
                }
                if (recfile && recfile->open_recordfile(true)) {
                    for (int i=0;i<rreq->record_source_indices_size();i++) {
                        auto rec = make_shared<Record>();
                        rec->set_batch_id(rreq->batch_id());
                        unsigned int src_idx = rreq->record_source_indices(i);
                        unsigned int rsize = recfile->size();
                        if (rsize>src_idx) {
                            record_data data = recfile->at(src_idx);
                            if (data.start!=nullptr && data.end>data.start) {
                                ptrdiff_t len = (ptrdiff_t) (data.end-data.start);
                                if (len>0 && len<netbatch.max_record_size) {
                                    rec->set_data(data.start, len);
                                } else {
                                    rec->set_error_code(ErrorCode::TOO_LARGE);
                                }
                            } else {
                                rec->set_error_code(ErrorCode::OTHER_ERROR);
                            }
                        } else {
                            rec->set_error_code(ErrorCode::INDEX_OUT_OF_BOUNDS);
                        }
                        if (rreq->record_indices_size()>i) {
                            rec->set_record_index(rreq->record_indices(i));
                        } else {
                             rec->set_record_index(i);
                        }

                        netbatch.response_queue.push(rec);
                    }
                } else {

                    for (int i=0;i<rreq->record_source_indices_size();i++) {
                        auto rec = make_shared<Record>();
                        rec->set_batch_id(rreq->batch_id());
                        if (rreq->record_source_indices_size()>i) {
                            rec->set_record_index(rreq->record_source_indices(i));
                        } else {
                             rec->set_record_index(i);
                        }
                        if (recfile) {
                            rec->set_error_code(ErrorCode::FILE_OPEN_FAILED);
                            cerr << "Failed to open " << recfile->basepath << endl;
                        } else {
                            rec->set_error_code(ErrorCode::INVALID_PATH);
                        }
                        netbatch.response_queue.push(rec);
                    }
                }

            } else {
                // We return one record for each record index
                for (int i=0;i<rreq->record_source_indices_size();i++) {
                    auto rec = make_shared<Record>();
                    rec->set_batch_id(rreq->batch_id());
                    if (rreq->record_source_indices_size()>i) {
                        rec->set_record_index(rreq->record_source_indices(i));
                    } else {
                         rec->set_record_index(i);
                    }
                    rec->set_error_code(ErrorCode::NOT_IMPLEMENTED);
                    netbatch.response_queue.push(rec);
                }

            }
            {
                unique_lock<mutex> lock_(netbatch.server_mutex);
                breq->completed_requests++;
                if (breq->completed_requests>=breq->batch_request.record_requests_size()) {
                        netbatch.requested_batches[rreq->batch_id()] = nullptr; // Ensures it gets deleted, unless we have another owner..
                }
            }
        }

};

class BatchRecordBroadcaster {
    nn::socket *sock;
    NetbatchServer &netbatch;

public:
    BatchRecordBroadcaster(NetbatchServer &netbatch) : sock(nullptr),netbatch(netbatch) {
    }

    void start() {
        sock = new nn::socket(AF_SP, NN_PUB);
        int nodelay = 1;
        sock->setsockopt(NN_TCP, NN_TCP_NODELAY, (char*) &nodelay,
                         sizeof (nodelay));
        int sndbufsize = 1024*1024*256;
        sock->setsockopt(NN_SOL_SOCKET, NN_SNDBUF, (char*) &sndbufsize,
                         sizeof (sndbufsize));

        sock->bind(netbatch.broadcast_url.c_str());
    }

    // Make this callable
    void operator()() {
        broadcast();
    }

    void broadcast() {
        while(1) {
            bool ser_success = false;
            {
                auto rec = netbatch.response_queue.pop();
                size_t len = rec->ByteSize();
                void *nnbuf = nn_allocmsg(len, 0);
                //void *nnbuf = malloc(len);
                if (nnbuf!=nullptr) {
                    rec->SerializeWithCachedSizesToArray((::google::protobuf::uint8*) nnbuf);
                    sock->send(&nnbuf, NN_MSG, 0); // Deallocates buffer after send, zero-copy op
                    //sock->send(nnbuf, len, 0);
                    //free(nnbuf);
                    //cout << "Sent " << len << " bytes " << endl;
                    if (rec->error_code()!=ErrorCode::OK) {
                        cout << "Error Packet Debug " << rec->DebugString() << endl;
                    }
                } else {
                    cerr << "CRITICAL: Send buffer allocation failed" << endl;
                }
            }
        }
    }

};


static NetbatchServer nb;

void request_acceptor() {
    BatchRequestAcceptor bra(nb);
    bra.start();
}

void request_consumer() {
    RecordRequestWorker brc(nb);
    brc.start();
}

static unsigned int rqize = 0;

int main(int argc, const char *argv[])
{
    ArgumentParser parser;

    // add some arguments to search for
    parser.addArgument("-u", "--url", 1, false);
    parser.addArgument("-b", "--broadcast_url", 1, false);
    parser.addArgument("-n", "--nthreads", 1, true);
    parser.addFinalArgument("basedir");

    // parse the command-line arguments - throws if invalid format
    parser.parse(argc, argv);

    // if we get here, the configuration is valid
    nb.server_url = parser.retrieve<string>("url");
    nb.broadcast_url = parser.retrieve<string>("broadcast_url");

    nb.set_basepath(parser.retrieve<string>("basedir"));
    nb.max_record_size = 1024*1024*10;
    int nthreads = atoi(parser.retrieve<string>("nthreads").c_str());
    if (nthreads<1 || nthreads>100) {
        cerr << "Number of threads set to invalid value. Must be from 1 to 100" << endl;
        return -1;
    }
    {
        cout << "Starting netbatch server" << endl
             << "\tServing from " << nb.basepath << endl
             << "\tAccepting requests at " << nb.server_url << endl
             << "\tBroadcasting batches at " << nb.broadcast_url << endl;
    }
    vector<std::thread*> worker_threads;
    for (int i=0;i<nthreads;i++) {
        worker_threads.push_back(new thread(request_consumer));
    }
    std::thread request_acceptor_thread(request_acceptor);

    BatchRecordBroadcaster broadcaster(nb);
    broadcaster.start();
    vector<thread> broadcasters;
    for (int i=0;i<nthreads-1;i++) {
        worker_threads.push_back(new thread(broadcaster));
    }
    broadcaster();
    return EXIT_SUCCESS;
}
