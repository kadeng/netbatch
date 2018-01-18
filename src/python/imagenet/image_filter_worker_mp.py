import io
import time
import PIL
import nnpy
import torchvision.transforms as transforms

import message_pb2 as msg
from imagenet.image_transforms import pil_to_numpy, numpy_hwc_to_chw, byte_to_float, RandomRotate

def worker_main():

  def normimg(img):
    img[0,:,:] = (img[0,:,:]-0.485) / 0.229
    img[1, :, :] = (img[0, :, :] - 0.456) / 0.224
    img[2, :, :] = (img[0, :, :] - 0.406) / 0.225
    return img


  sub = nnpy.Socket(nnpy.AF_SP, nnpy.PULL)
  sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVBUF, 1024*1024*300)
  sub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
  sub.setsockopt(nnpy.SOL_SOCKET, nnpy.RCVTIMEO, 1000*10)
  sub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)

  #sub.connect('tcp://144.76.34.134:9877')
  sub.connect('tcp://127.0.0.1:9877')
  pub = nnpy.Socket(nnpy.AF_SP, nnpy.PUSH)
  pub.setsockopt(nnpy.SOL_SOCKET, nnpy.SNDBUF, 1024*1024*300)
  pub.setsockopt(nnpy.SOL_SOCKET, 16, 1024*1024*300)
  #pub.setsockopt(nnpy.TCP, nnpy.TCP_NODELAY, 1)
  #pub.connect('tcp://127.0.0.1:9878')
  pub.connect('ipc:///tmp/imgpipe.sock')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



  img_transforms = transforms.Compose([
            RandomRotate(-3, 2),
            transforms.Scale(244),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            pil_to_numpy,
            #numpy_to_hsv,
            numpy_hwc_to_chw,
            byte_to_float,
            normimg
        ])

  start = time.time()
  startrec = time.time()
  i = 0
  nbytes = 0
  okcount = 0
  print("Working ...")
  while(True):
    while True:
        try:
            recd = sub.recv()
            break
        except nnpy.NNError as e:
            if (e.error_no==nnpy.ETIMEDOUT):
                print("Nothing to do")
            else:
                print("Exception: %r - %d" % (e, e.error_no))
    #print("Recv %d bytes" % (len(recd)))
    nbytes += len(recd)
    rec = msg.Record()
    rec.ParseFromString(recd)
    i+=1
    if (rec.error_code==msg.OK):
        try:
            image = PIL.Image.open(io.BytesIO(rec.data))
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

if __name__=='__main__':
  import multiprocessing
  import sys
  import time
  processes = int(sys.argv[1])
  print("Launching %d processes" % (processes))
  time.sleep(2.0)
  for i in range(processes):
    print("Launching Worker %d" % (i))
    d = multiprocessing.Process(target=worker_main, name='worker_%d' % (i))
    d.daemon=True
    d.start()
    time.sleep(0.3)
  while(True):
    time.sleep(5.0)
    print("Press CTRL-C to stop")
