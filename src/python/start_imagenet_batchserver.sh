#!/bin/bash
batchserver --url tcp://*:9876 --broadcast_url tcp://*:9877 --nthreads 6 /data/datasets/imagenet/imagenet-data/train-recs/
