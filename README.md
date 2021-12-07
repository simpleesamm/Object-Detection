# Object-Detection
General Repository for Yolov4_Darknet & Tensorflow Object Detction

This is my main object detection pipeline where you can train a model from scratch with darknet and further port it to Tensorflow if needed. 

**Credits: **
The main backbone of these codes are by theAIGuysCode & Roboflow.
https://github.com/theAIGuysCode/tensorflow-yolov4-tflite (For Porting to Tensorflow)
https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/ (For Model Training)

**Instructions & Files:**
1) Start with the model training (Model_Training.ipynb). I ran this on Google Colab to use their GPU. 

2) After training it is important to download the model weights to convert it to a Tensorflow Model in the future.

3) Setting up Tensorflow Model
You can opt to follow theAIGuysCode throught the link above.
- Environment Creation
```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```
- Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

- Converting the Darknet weights to Tensorflow Weights
```
# the "--" is user inputs that you would give to the command prompt
# you want to change the yolov4.weights to the name of your weight
# your model will be saved under "checkpoints/yolov4-416"
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
```
4) Run the model!
```
# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```
_Run On My Formats_
- detectEditedCSV.py runs the model through your selected folder of images and outputs them in a CSV format (Great for Kaggle Submissions)
- watchdog_image2.py runs the model continuously, watching a particular folder for new images and outputting the results each time. 

You can find output detection in the detections folder.
```
#CSV Output
python detectEditedCSV.py

#Watch a folder continuously
python watchdog_image2.py
```

Note: For my detection formats, I integrated a form of TTA (Time Test Augmentation) code to further improve the accuracy of the models. Howevever instead of the collective accuracy score it just goes by a majority voting scheme. I do think that it can be easily edited to be a collective accuracy format. 

This would also mean that for every image you would be running the model 5 times due to TTA so don't use this for Real Time stuff. 
