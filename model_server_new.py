import json 
from flask import jsonify
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
from flask import Flask
from flask import render_template, request
from werkzeug.utils import secure_filename
import base64
import io
app = Flask(__name__)

print("Test started")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
input_size = 416
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
print("Model loaded")
monitor_dir = r'C:/Yolo2Tensor/tensorflow-yolov4-tflite/data/images/test'
collageDirectory = r'C:/Yolo2Tensor/tensorflow-yolov4-tflite/collage'
UPLOAD_FOLDER = 'C:/Users/yunxing/Downloads/Telegram Desktop/YoloTensor2/Yolo2Tensor/tensorflow-yolov4-tflite/upload_folder/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        print('post request received')
        label = ''
        file = request.json.get('photo')
        # filename = secure_filename(file.filename)
        # base_path = os.path.dirname(__file__)
        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(image_path)
        imgdata = base64.b64decode(file)
        PILImage = Image.open(io.BytesIO(imgdata)).rotate(180)
        #print(PILImage)
        print("image received")
        label = run_model(PILImage)
        return label
    return render_template('index.html')
    

count = 1
imageList = []
    # loop through images in list and run Yolov4 model on each
def run_model(PILImage):
    countFilesInCollage = len([name for name in os.listdir(collageDirectory) if os.path.isfile(os.path.join(collageDirectory, name))])
    if countFilesInCollage == 0:
        imageList.clear()
    global count
    # print(r'{}'.format(image_path))

    # time.sleep(1)
    # #original_image = plt.imread(r'{}'.format(image_path.replace('\\', '/')))
    # original_image = cv2.imread(r'{}'.format(image_path.replace('\\', '/')))
    # print(original_image)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)



    original_image = cv2.cvtColor(np.array(PILImage), cv2.IMREAD_COLOR)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
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
            score_threshold=0.25
        )

    #Get scores & classes and sort them into finalDictionary (not ranked based on max yet)
    scoreArray = (scores.numpy()).flat
    classesArray = (classes.numpy()).flat
    finalScoreArray = []
    finalClassArray = []
    finalDictionary = {}

    listOfClasses = ["A", "B", "Bullseye", "C", "D", "Down", "E", "F", "G", "H", "Left", "Right", "S", "Stop",
                     "T", "U", "Up", "V", "W", "X", "Y", "Z", "eight", "five", "four", "nine",
                     "one", "seven", "six", "three", "two"]
                     
    listOfIDs = [15, 16, 31, 17, 18, 2, 19, 20, 21, 22, 4, 3, 23, 5, 24, 25, 1, 26, 27, 28, 29, 30, 13, 10, 9, 14, 6, 12, 11, 8, 7]

    listOfRequiredClasses = ["Alphabet A", "Alphabet B", "Bullseye", "Alphabet C", "Alphabet D", "down arrow", "Alphabet E", "Alphabet F", "Alphabet G", "Alphabet H", "left arrow", "right arrow", "Alphabet S", "Stop",
                     "Alphabet T", "Alphabet U", "Up arrow", "Alphabet v", "Alphabet w", "Alphabet x", "Alphabet y", "Alphabet z", "eight", "five", "four", "nine",
                     "one", "seven", "six", "three", "two"]
    if scoreArray[0] > 0:

        for i in range(len(scoreArray)):
            if scoreArray[i] > 0.0:
                finalScoreArray.append(scoreArray[i])
                finalClassArray.append(classesArray[i])
                finalDictionary.update({listOfClasses[int(finalClassArray[i])]: str(scoreArray[i])})



    else:
        print("No Detections")

    print("finalDictionary: ", finalDictionary)

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
    print(class_names)

    allowed_classes = list(class_names.values())

    print(allowed_classes)
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))

        
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    imageList.append(image)
    imagecollage = cv2.hconcat(imageList)

    cv2.imwrite('./detections/' + 'detection' + str(count) + '.jpg', image)
    cv2.imwrite('./collage/' + 'detection' + str(count) + '.jpg', imagecollage)
    count += 1
    print(finalDictionary)
    return jsonify(finalDictionary)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0')
