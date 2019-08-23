import numpy as np
import pandas as pd
import caffe

import cv2

import os

import skimage.data
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']


def load_detectors():
    mean = np.load('./gnetmodels/ilsvrc_2012_mean.npy').mean(1).mean(1)

    #pretrained_model = "./gnetmodels/rcnn/bvlc_reference_rcnn_ilsvrc13.caffemodel"
    pretrained_model = "./gnetmodels/referencecaffenet/bvlc_reference_caffenet.caffemodel"
    #model_def = "./gnetmodels/rcnn/deploy.prototxt"
    model_def = "./gnetmodels/referencecaffenet/deploy.prototxt"

    raw_scale = 255.0

    channel_swap = [2,1,0]#'2,1,0'
    context_pad = 16

    output_file = "someoutput.h5"

    global detector
    detector = caffe.Detector(model_def, pretrained_model, mean=mean,
                input_scale=None, raw_scale=raw_scale,
                channel_swap=channel_swap,
                context_pad=context_pad)
    return detector

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    mean, channel_swap = None, None
    mean = np.load('./gnetmodels/ilsvrc_2012_mean.npy').mean(1).mean(1)
    
    crop_mode="selective_search"
    choices = CROP_MODES
    pretrained_model = "./gnetmodels/rcnn/bvlc_reference_rcnn_ilsvrc13.caffemodel"
    model_def = "./gnetmodels/rcnn/deploy.prototxt"

    raw_scale = 255.0

    channel_swap = [2,1,0]#'2,1,0'
    context_pad = 16

    output_file = "someoutput.h5"

    detector = caffe.Detector(model_def, pretrained_model, mean=mean,
                input_scale=None, raw_scale=raw_scale,
                channel_swap=channel_swap,
                context_pad=context_pad)

    inputs = ['./gnetmodels/one_car.jpg']
    #inputs = pd.read_csv("./test.csv", sep=',', dtype={'filename':str})
    #inputs.set_index('filename', inplace=True)
  
    img = cv2.imread('./gnetmodels/one_car.jpg')
    width, height, channels = img.shape
    #data = [{'filename':'/home/oscar/Dokument/Development/video_code/coddetection/gnetmodels/one_car.jpg',
    #        'xmin':0,'ymin':0,'xmax':width,'ymax':height}]
    
    images_windows = [('/home/oscar/Dokument/Development/video_code/coddetection/gnetmodels/one_car.jpg', 
            np.array([[0,0,width,height]]))] 
    #inputs = pd.DataFrame(data)
    #inputs.set_index('filename', inplace=True)

    #images_windows = [(ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
    #                    for ix in inputs.index.unique()]
    
    print(type(images_windows))
    print(images_windows)

    detections = detector.detect_windows(images_windows)
    #images_windows = [(ix, inputs[np.where(inputs.index == ix)][COORD_COLS].values())
    #                    for (ix,val) in enumerate(inputs)]
    #images_windows = [(ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values())
    #                    for ix in inputs.index.unique()]
    #detections = detector.detect_windows()
    df = pd.DataFrame(detections)
    df.set_index('filename', inplace=True)
    df[COORD_COLS] = pd.DataFrame(data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)
    del(df['window'])

def process_image(img):
    from skimage.segmentation import slic
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float
    from skimage import io
    
    image = img_as_float(io.imread(img))
    

    segments = slic(image, n_segments=50, sigma=6)
    for (i, segVal) in enumerate(np.unique(segments)):
        image[segments == segVal] = np.mean(image[segments == segVal])
    
    fig = plt.figure("Superpixels, n_segments=numSegments, sigma=5")
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image)
    plt.axis('off')
    plt.show()

    '''for numSegments in (100, 200, 300):

        segments = slic(image, n_segments=numSegments, sigma=5)

        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis('off')
    plt.show()'''
    return image

def second_main(argv):
    img = skimage.io.imread('./gnetmodels/cars.jpg')
    #super_img = process_image('./gnetmodels/cars.jpg') 
    super_img = img
    load_detectors()
    '''best_s=1000
    best_sig=1.0
    best_min_size=100
    best_min_elapsed_time=1000000
    for s in np.arange(100,1000,100):# range(100, 1000, 100):
        for sig in np.arange(0.1,0.9,0.1):# range(0.1,0.9,0.1):
            for m in np.arange(10,100,10):# range(10,100,10):
                start_time = time.time()
                img_lbl, regions = selectivesearch.selective_search(img, scale=s, sigma=sig, min_size=m)
                elapsed_time = time.time() - start_time
                #print("Using scale: ",s, " sigma: ", sig, " min_size: ", m)
                #print("Elapsed time: ", elapsed_time)
                if (elapsed_time < best_min_elapsed_time):
                    best_s = s
                    best_sig = sig
                    best_min_size=m
                    best_min_elapsed_time = elapsed_time
                
    print("Found: ")
    print("Best scale: ", best_s)
    print("Best sigma: ", best_sig)
    print("Best min_size: ", best_min_size)
    print("Best elapsed_time: ", best_min_elapsed_time)'''


    start_time = time.time()
    img_lbl, regions = selectivesearch.selective_search(super_img, scale=1000, sigma=0.9, min_size=80)
    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)
    candidates = set()

    for r in regions:
        if (r['size'] in candidates):
            continue
        if(r['size'] < 5000):
            continue

        x,y,w,h = r['rect']
        if (w/h > 1.2 or h/w > 1.2):
            continue
        if (x+w >= 500 and y+h >= 300):
            continue
        candidates.add(r['rect'])

    start_time = time.time()
    evaluate_the_regions(img, candidates)
    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)

    fid, ax = plt.subplots(ncols=1,nrows=1,figsize=(6,6))
    ax.imshow(img)

    for x,y,w,h in candidates:
        #print(x,y,w,h)
        rect = mpatches.Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1)
        ax.add_patch(rect)

    plt.show()

def evaluate_the_regions(img, candidates):
    temp_file = "./temp_image.png"
    skimage.io.imsave(temp_file, img)
   
    images_windows = []
    for x,y,w,h in candidates:
        images_windows.append((temp_file, np.array([[x,y,x+w,y+h]])))

    full_detections = []
    for i in images_windows:
        #print(i)
        try:
            detections = detector.detect_windows([i])
            full_detections.append((i, detections))
            #print("Detections")
            #print(detections)
        except ValueError:
            #print("Got a error")
            pass

    #print(full_detections)
    return full_detections

if __name__ == "__main__":
    import sys
    #main(sys.argv)
    second_main(sys.argv)
    #process_image('./gnetmodels/car_queue1.jpg')
