import threading
import atexit
import signal

import numpy as np
import pandas as pd
import caffe

import cv2
import os
import skimage.data

import selectivesearch
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

# These import are for the preprocessing step
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# Use this library to draw the rectangles
from PIL import Image,ImageDraw

#
#   The InformationBase keeps track of all the resources provided by the detectors.
#   Reosurces such as latest image or angles can be retrieved by calling the 
#   functions get_angle_information_for or get_latest_image_for.
#
class InformationBase():
    def __init__(self, number_of_detectors):
        self.angles = {}
        self.images = []
        for n in range(number_of_detectors):
            self.angles[n] = (0.0, [])
            self.images.append((0.0, None))
       
        self.lock = threading.Lock()

    #
    #   camera_id is an integer refering to a camera.
    #   information is a list containing the angles for a found sub_image.
    #  
    #   This function is frequently called by the detectors to provide the
    #   program with information about what it detects
    #
    def update_angle_information(self, camera_id, information):
        self.lock.acquire()
        self.angles[camera_id] = information 
        self.lock.release()

    #
    #   camera_id is an intenger refering to a camera.
    #   image is the last image that the detector took a snapshot of.
    #
    #   This function should be frequently called by the detectors to provide
    #   the program with as recent images as possible.
    #
    def update_image_information(self, camera_id, image):
        self.lock.acquire()
        self.images[camera_id] = image
        self.lock.release()

    #
    #   camera_id is an integer refering to a camera.
    #
    #   Use this function to get the latest information about a angle.
    #
    def get_angle_information_for(self, camera_id):
        information = []
        self.lock.acquire()
        information = self.angles[camera_id]
        self.lock.release()

    #
    #   camera_id is an integer refering to a camera.
    #
    #   Use this function to get the latest image containing all detected
    #   angles.
    #
    def get_latest_image_for(self, camera_id):
        image = None
        self.lock.acquire()
        image = self.images[camera_id]
        self.lock.release()

#
#   This class translates found classes to objects that we think is of significance.
#   Pass a related_word.txt with all classes separated by new lines.
#
class WordDetector():
    def __init__(self, related_words_path):
        self.related_words = set()
        words = open(related_words_path, "r")
        for word in words:
            self.related_words.add(word.strip())
        words.close()
        self.lock = threading.Lock()

    def is_related_word(self, word):
        self.lock.acquire()
        is_related = True if (word in self.related_words) else False
        self.lock.release()
        return is_related

# 
#   The detectors are defined as a ClassificationThreads. All detectors share resources thorugh the 
#   information_base. Information such as last processed image or angles for found vehicles are passed 
#   the information_base.   
#
class ClassificationThread(threading.Thread):
    def __init__(self, camera_id,
                 mean, pretrained_model, 
                 model_def, raw_scale,channel_swap,
                 context_pad,
                 synset_file,
                 related_word_detector,
                 information_base,
                 process_images=False,
                 mode_gpu=False):
        threading.Thread.__init__(self)

        self.stop_var = False

        self.camera_id = camera_id
        self.process_images = process_images
        
        self.mean = np.load(mean).mean(1).mean(1)
       
        # Enable GPU computations with the mode_gpu=True. This is only possible if caffe are
        # compiled with CUDA. By default it is set to use the CPU.
        if (mode_gpu):
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        # Setup the detector
        self.detector = caffe.Detector(model_def, pretrained_model,
                                       mean=self.mean, input_scale=None,
                                       raw_scale=raw_scale,
                                       channel_swap=channel_swap,
                                       context_pad=context_pad)
       
        self.classifier = caffe.Classifier(model_def, pretrained_model, 
                                           image_dims=None,mean=self.mean,
                                           input_scale=None, 
                                           raw_scale=raw_scale,
                                           channel_swap=channel_swap)
        
        self.net = caffe.Net(model_def,
                             pretrained_model, 
                             caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', self.mean)
        self.transformer.set_raw_scale('data', raw_scale)
        self.transformer.set_channel_swap('data', channel_swap)
        
        # Parse the labels for the classifications
        f = open(synset_file)
        self.labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        f.close()
        self.labels_df.sort_values(by='synset_id')

        self.labels = np.loadtxt(synset_file, str, delimiter='\t')
        self.related_word_detector = related_word_detector
        self.informationbase = information_base

        # Initialize the video capture
        # self.capturer = cv2.VideoCapture(self.camera_id)
        
    def __del__(self):
        # Delete the capturer
        # self.capturer.release() 
        pass

    # This functions applys SLIC and automatically segments the picture and calculates a
    # mean of each region.
    def process_image(self, image):
        
        image = img_as_float(skimage.io.imread(image))
    
        segments = slic(image, n_segments=50, sigma=6)
        for (i, segVal) in enumerate(np.unique(segments)):
            image[segments == segVal] = np.mean(image[segments == segVal])
        
        return image

    def classify_image(self, regions, path_temp_file):

        images_windows = []
        for x,y,w,h in regions:
            images_windows.append((temp_file, np.array([[x,y,x+w,y+h]])))

        full_detections = []
        try:
            detections = self.detector.detect_windows(images_windows)
            full_detections.append(detections)
        except ValueError:
            pass

        if (len(full_detections) > 0):
            something_found, found_vehicles = self.image_contains_vehicles(full_detections)
        return full_detections
    #
    # This is a function for checking if any of the detections are including a vehicle
    #
    def image_contains_vehicles(self, detections):
        # Merge the two lists together
        names = self.labels_df['name'].tolist()
        names_predictions = list(zip(names, detections))
      
        # Sort the predictions with the one with highest probability first
        names_predictions = sorted(names_predictions, key=lambda n : n[1], reverse=True)
       
        print(names_predictions)
        # Debug output, this show how many 
        # print("Top 3 predictions:")
        # print("1. ", names_predictions[0])
        # print("2. ", names_predictions[1])
        # print("3. ", names_predictions[2])

        label = names_predictions[0][0] if (self.related_word_detector.is_related_word(names_predictions[0][0])) else ""
        if (label == ""):
            label = names_predictions[1][0] if (self.related_word_detector.is_related_word(names_predictions[1][0])) else ""
        if (label == ""):
            label = names_predictions[2][0] if (self.related_word_detector.is_related_word(names_predictions[2][0])) else ""
       
        return False if (label == "") else True, label

    #
    #   Find potential objects in the image. Then evaluate if they are vehicles or not bys using the
    #   evaluate_detectoins function.
    #
    def find_potential_regions(self, img, img_path):
        # A crucial part of the program. It performs the selective search rather than a sliding window
        # search. All areas that are found are stored in the regions structure.
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.5, min_size=20)
        
        # Used for filtering only.
        candidates = []

        # Keep any region of proper size
        # Consider to remove detections for the full picture
        coordinates = set()
        for r in regions:
            if (r['rect'] in coordinates):
                continue
            if (r['size'] < 2000):
                continue
            
            x,y,w,h = r['rect']
            if (w/h > 1.2 or h/w > 1.2):
                continue
            print("Added somehting: ", x , " ", y, " ", x+w, " ", y+h)
            coordinates.add(r['rect'])
            candidates.append((img, np.array([[x,y,x+w,y+h]])))
        
        # Last, evaluate if they are vehicles or somehting else
        detected_vehicles = self.evaluate_detections(img, candidates)

        # Then return the found vehicles
        return detected_vehicles

    #
    #   A function that runs through each sub_image from the proposed image_windows. 
    #
    def evaluate_detections(self, img, images_windows):
        total_vehicles_found = []
        for sub_image in images_windows:
            try:
                # Start by doing a prediction of what i might be
                #detections = self.detector.detect_windows([sub_image])
                #detections = self.classifier.predict([img], oversample=False)
                transformed_image = self.transformer.preprocess('data', 
                                                                img[sub_image[1][0][0]:sub_image[1][0][2]]
                                                                   [sub_image[1][0][1]:sub_image[1][0][3]])
                self.net.blobs['data'].data[...] = transformed_image

                output = self.net.forward()

                # This function picks the class that has the highest probability.
                o = output['prob'][0].argmax()

                print(o)
                #break
                #Second, check if the detections matches a vehicle
                #something_found, label = self.image_contains_vehicles(detections[0]['prediction'])
                #break
                something_found = True
                label = "alfred"
                if (something_found):
                    # This means that we found a vehicle. Therefore, the sub_image should be cosidered as
                    # an important area.
                    total_vehicles_found.append((label, sub_image))
                    #print("Found : ", label)
                if (self.stop_var):
                    break
            except ValueError:
                # Silently ignore each error that occurs, such as matrix dimension errors.
                pass
        return total_vehicles_found

    def calculate_angles(self, vehicles, orig_img_dimensions):

        calculated_angles = []
        x=orig_img_dimensions[0]
        y=orig_img_dimensions[1]
        center_of_image=x/2
        max_angle=25
        for vehicle in vehicles:
            
            x0 = vehicle[1][1][0][0]
            x1 = vehicle[1][1][0][2]
            center_of_vehicle=(x0+x1)/2
            if center_of_vehicle==center_of_image:
                degree=0
            else:
                degree=(((center_of_vehicle-center_of_image)*max_angle)/center_of_image)
                
            calculated_angles.append((vehicle, degree))

        return calculated_angles

    def run(self):
        # TODO Implement so that the camera is reading instead of a single picture 

        while (True):
            img = skimage.io.imread('./gnetmodels/cars.jpg')
            #img = [caffe.io.load_image('./gnetmodels/cars.jpg')]

            eval_img = img
        
            processed_image = None
            if (self.process_images):
                processed_image = self.process_image('./gnetmodels/cars.jpg')
                eval_img = processed_image

            # Classify the image with something
            # Either just look for regions or try to decide of there are any cars at all
            detected_vehicles = self.find_potential_regions(eval_img, './gnetmodels/cars.jpg')

            # 
            save_img = Image.fromarray(eval_img)
            draw_save_image = ImageDraw.Draw(save_img)
            
            # Paint the detected vehicles
            for vehicle in detected_vehicles:
                #print("I saw a ", vehicle[0])
                # The format is: 
                #        vehicle[1] = sub_image
                #        vehicle[1][1] = coordinate_array
                #        vehicle[1][1][0][0] = access x0
                #        vehicle[1][1][0][1] = access y0
                #        vehicle[1][1][0][2] = access x1
                #        vehicle[1][1][0][3] = access y1
                print(vehicle)
                draw_save_image.line([vehicle[1][1][0][0], 
                                      vehicle[1][1][0][1],
                                      vehicle[1][1][0][0],
                                      vehicle[1][1][0][3]],
                                      fill=128)
                draw_save_image.line([vehicle[1][1][0][0], 
                                      vehicle[1][1][0][1],
                                      vehicle[1][1][0][2],
                                      vehicle[1][1][0][1]],
                                      fill=128)
                draw_save_image.line([vehicle[1][1][0][2],
                                      vehicle[1][1][0][1],
                                      vehicle[1][1][0][2],
                                      vehicle[1][1][0][3]],
                                      fill=128)
                draw_save_image.line([vehicle[1][1][0][0],
                                      vehicle[1][1][0][3],
                                      vehicle[1][1][0][2],
                                      vehicle[1][1][0][3]],
                                      fill=128)

            # TODO: Find the angles and update the information base
            angles = self.calculate_angles(detected_vehicles, eval_img.shape)
            self.informationbase.update_angle_information(self.camera_id, angles)
            
            print("Angles:")
            print(angles)

            # Update image in the information base
            save_img.show()
            self.informationbase.update_image_information(self.camera_id, save_img)
            
            # Use this to stop each thread.
            if (self.stop_var):
                break

            # Just for testing purposes. It makes the detector to run just once.
            break

    def stop(self):
        print("Terminates classifier")
        self.stop_var = True

class ThreadedDetector():
    def __init__(self):
        
        atexit.register(self.safe_exit)
        #signal.signal(signal.SIGINT, safe_exit)
        #signal.signal(signal.SIGTERM, safe_exit)

        print("Init information base")
        self.information_base = InformationBase(1)

        print("Init related objects detector")
        self.related_word_detector = WordDetector('./gnetmodels/car_related_words.txt')

        print("Init classificationthreads")

        self.camera_detectors = []

        # TODO: Add a more general way to add cameras, maybe a config file.

        # Set the camera_id to be the camera on which the detector should capture
        self.camera_detectors.append(ClassificationThread(camera_id=0,
                                                    mean='./gnetmodels/ilsvrc_2012_mean.npy',
                                                    pretrained_model='./gnetmodels/googlenet/bvlc_googlenet.caffemodel',
                                                    model_def='./gnetmodels/googlenet/deploy.prototxt',
                                                    raw_scale=255,
                                                    channel_swap=[2,1,0],
                                                    context_pad=16,
                                                    synset_file='./gnetmodels/synset_words.txt',
                                                    information_base=self.information_base, 
                                                    related_word_detector=self.related_word_detector))

        # This is a tiresome task but just copy the code above and state which arguments it should have.

        print("The detectors are initialized...")

    # Use this function when the Threadeddetector should be runned as a standalone program.
    def start(self):
        # Start detecting!
        for i in self.camera_detectors:
            i.start()

        # Enable the user to terminate the program
        while(True):
            var = raw_input("\nPress q to terminate...\n")
            if (var == 'q'):
                self.safe_exit()
                break

    # Use this function if building somehting like a client.
    def _start(self):
        for i in self.camera_detectors:
            i.start()

    # Use this function to stop the detectors.
    def _stop(self):
        self.safe_exit()

    # This function stops the detctors in a safe way.
    def safe_exit(self):
        print("Terminates the the classifiers...")
        for detector in self.camera_detectors:
            detector.stop()

# Uncomment this either for debugging or for running the Threadeddetector as a standalone program.
if __name__ == "__main__":
    import sys
    threaded_detector = ThreadedDetector()
    threaded_detector.start()

