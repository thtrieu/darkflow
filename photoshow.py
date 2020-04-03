# importing the dependencies
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import argparse

# the cfg file and weights location change this thing according to your model location
model = {"model": "cfg/yolov2.cfg",
           "load": "yolov2.weights",
           "threshold": 0.4}

# creating the object
tfnet = TFNet(model)

# to get the image path
parser = argparse.ArgumentParser()
parser.add_argument('--img',type=str,help='path of the image')
arg = parser.parse_args()

imgcv = cv2.imread(arg.img) # read the image
result = tfnet.return_predict(imgcv) # predict the classes and cordinates of the oject

# This is for draw the bounding box around the predicted classes 
tl = []
br = []
labels = []
for i in range(len(result)):
    topleft = (result[i]['topleft']['x'],result[i]['topleft']['y']) # to get the labels from the predicted class ,it's in the form of dictionary
    bottomright = (result[i]['bottomright']['x'],result[i]['bottomright']['y'])
    label = (result[i]['label'])
    st = result[i]['topleft']['x'] 
    nd = result[i]['bottomright']['x']
    mid_x = (nd-st)//2 + st # mid point of the top box line
    mid_y = result[i]['topleft']['y']
    tl.append(topleft)
    br.append(bottomright)
    labels.append(label)
    img2 = cv2.rectangle(imgcv,tl[i],br[i],(0,255,255),5) # draw rectangles around the classes here we pass image,topleft cordinates ,bottomright cordinates ,which colour box we want and how thik the line
    img2 = cv2.putText(imgcv,labels[i],tl[i],cv2.FONT_HERSHEY_COMPLEX,1, (0 ,0 ,0), 2) # putting the label on the topleft corner
    img2 = cv2.putText(imgcv,confidence[i],cf_cor[i],cv2.FONT_ITALIC,1, (0 ,0 ,255), 2) # putting the confidence score


    
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)  # convert the image in RGB format
cv2.imshow('prediction',img2)
cv2.waitKey(0) # waitkey for hold the image in display until user press any key
