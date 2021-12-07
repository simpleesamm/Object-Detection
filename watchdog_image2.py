import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import time 
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt
import collections

print("Test started 1")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
input_size = 416
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])


print("Model loaded")
monitor_dir = r'C:/Yolo2Tensor/tensorflow-yolov4-tflite/data/images/test'
collageDirectory = r'C:/Yolo2Tensor/tensorflow-yolov4-tflite/collage'



imageList = []




count = 1
    # loop through images in list and run Yolov4 model on each
def run_model(image_path):
    countFilesInCollage = len([name for name in os.listdir(collageDirectory) if os.path.isfile(os.path.join(collageDirectory, name))])
    if countFilesInCollage == 0:
        imageList.clear()

    originalClass = []
    originalScore = []


    brightClass = []
    brightScore = []


    darkClass = []
    darkScore = []


    sharpenClass = []
    sharpenScore = []


    gBlurClass = []
    gBlurScore = []


    blurClass = []
    blurScore = []

    storeBoxes = []
    storeScores = []
    storeClasses = []
    storeValidDetection = []


    global count
    print(r'{}'.format(image_path))

    time.sleep(2)
    #original_image = plt.imread(r'{}'.format(image_path.replace('\\', '/')))
    original_image = cv2.imread(r'{}'.format(image_path.replace('\\', '/')))
    #print(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    original = image_data / 255.

    #augmentation for TTA
    bright = np.ones(image_data.shape, dtype = "uint8")*70
    augBright = cv2.add(image_data,bright)
    augBright = augBright / 255.

    dark = np.ones(image_data.shape, dtype="uint8") * 70
    augDark = cv2.subtract(image_data,dark)
    augDark = augDark / 255.

    sharpening = np.array([[-1,-1,-1],
                          [-1,10,-1],
                          [-1,-1,-1]])

    sharpen = cv2.filter2D(image_data,-1,sharpening)
    sharpen = sharpen / 255.

    gaussianBlur = cv2.GaussianBlur(image_data,(3,3),cv2.BORDER_DEFAULT)
    gaussianBlur = gaussianBlur / 255.

    blur = cv2.blur(image_data,(3,3))
    blur = blur / 255.

    imageAug2dArr = [original,augBright,augDark,sharpen,gaussianBlur,blur]




    for x in range (6):
        images_data = []
        for i in range(1):
            images_data.append(imageAug2dArr[x])
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.3
            )
        # storeVisualizations
        storeBoxes.append(boxes.numpy())
        storeScores.append(scores.numpy())
        storeClasses.append(classes.numpy())
        storeValidDetection.append(valid_detections.numpy())

        #print("check scorenumpy ", classes.numpy())
        scoreArray = (scores.numpy()).flat
        classesArray = (classes.numpy()).flat

        finalScoreArray = []
        finalClassArray = []
        finalDictionary = {}

        listOfClasses = ["Alphabet A", "Alphabet B", "Bullseye", "Alphabet C", "Alphabet D", "down arrow", "Alphabet E", "Alphabet F", "Alphabet G", "Alphabet H", "left arrow", "right arrow", "Alphabet S", "Stop",
                         "Alphabet T", "Alphabet U", "Up arrow", "Alphabet v", "Alphabet w", "Alphabet x", "Alphabet y", "Alphabet z", "eight", "five", "four", "nine",
                         "one", "seven", "six", "three", "two"]

        listOfIDs = [15, 16, 31, 17, 18, 2, 19, 20, 21, 22, 4, 3, 23, 5, 24, 25, 1, 26, 27, 28, 29, 30, 13, 10, 9, 14, 6, 12, 11, 8, 7]

        if scoreArray[0] > 0 :

            for i in range(len(scoreArray)):
                if scoreArray[i] > 0.0:

                    if x == 0:
                        originalClass.append(classesArray[i])
                        originalScore.append(scoreArray[i])
                        print("OriginalClass: " , str(originalClass))
                    elif x == 1:
                        brightClass.append(classesArray[i])
                        brightScore.append(scoreArray[i])
                        print("brightClass: ", str(brightClass))
                    elif x == 2:
                        darkClass.append(classesArray[i])
                        darkScore.append(scoreArray[i])
                        print("darkClass: ", str(darkClass))
                    elif x == 3:
                        sharpenClass.append(classesArray[i])
                        sharpenScore.append(scoreArray[i])
                        print("sharpenClass: ", str(sharpenClass))
                    elif x == 4:
                        gBlurClass.append(classesArray[i])
                        gBlurScore.append(scoreArray[i])
                        print("gBlurClass: ", str(gBlurClass))
                    elif x == 5:
                        blurClass.append(classesArray[i])
                        blurScore.append(scoreArray[i])
                        print("blurClass: ", str(blurClass))

        else:
            print("No Detections")





    lstClass = [originalClass, brightClass, darkClass, sharpenClass, gBlurClass, blurClass]
    lstScore = [originalScore, brightScore, darkScore, sharpenScore, gBlurScore,blurScore]

    finalIndex = 0

    for i in range(int(len(lstClass)/2)):
        counter = 6
        checkImageCounter = 0
        for k in range(counter):
            if collections.Counter(lstClass[i]) == collections.Counter(lstClass[k]):
                print("The lists are identical")
                print(i, " vs ", k)
                checkImageCounter+=1
            else:
                print("The lists are not identical")
                print(i," vs ",k)

            counter-=1

        if checkImageCounter >= 3:
            print("Choose list ", i)
            finalIndex = i
            for j in range(len(lstScore[i])):
                finalScoreArray.append(lstScore[i][j])
                finalClassArray.append(lstClass[i][j])
                finalDictionary.update({listOfClasses[int(finalClassArray[j])]: lstScore[i][j]})

            break
        else:
            print("No distinct winners")

    print("finalDictionary: ", finalDictionary)


    #pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    print("finalIndex ", finalIndex)
    pred_bbox = [storeBoxes[finalIndex], storeScores[finalIndex], storeClasses[finalIndex], storeValidDetection[finalIndex]]

        # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))
    #imageTemp = image

        
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    imageList.append(image)



    imagecollage = cv2.hconcat(imageList)
    cv2.imwrite('./collage/' + 'detection' + str(count) + '.jpg', imagecollage)
    cv2.imwrite('./detections/' + 'detection' + str(count) + '.jpg', image)







    count += 1

# change it to your monitoring directory
class ExampleHandler(FileSystemEventHandler):
    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print("Got event for file %s" % event.src_path)
        run_model(event.src_path)
        print("recognition finished")

observer = Observer()
event_handler = ExampleHandler() # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path=monitor_dir)
observer.start()
print("watchdog started, waiting for new images")

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
