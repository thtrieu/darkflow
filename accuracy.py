

  
from collections import namedtuple
import numpy as np
import cv2,time,json
import unicodedata


  
from xml.dom import minidom
import numpy as np
t=time.time()

def get_box_from_xml(file):
 
    doc = minidom.parse(file)
    names=[]
    # doc.getElementsByTagName returns NodeList
    objects = doc.getElementsByTagName("object")
    lis=[]
    for object in objects:
        name=object.getElementsByTagName("name")[0].firstChild.data
        names.append(name)
        object=object.getElementsByTagName("bndbox")[0] 
        xmin=int(object.getElementsByTagName("xmin")[0].firstChild.data)
        ymin=int(object.getElementsByTagName("ymin")[0].firstChild.data)
        xmax=int(object.getElementsByTagName("xmax")[0].firstChild.data)
        ymax=int(object.getElementsByTagName("ymax")[0].firstChild.data)
        lis.append([xmin,ymin,xmax,ymax])
        
    return np.asarray(lis).reshape(-1,4),names

 

def get_box_from_json(file):
    
    with open(file) as f:
        boxes=json.load(f)
    
    lis=[]
    lis_name=[]
    

    for box in boxes:
        
        name=box["label"]
        name=unicodedata.normalize('NFKD', name).encode('ascii','ignore')

         
    for box in boxes:
        
        name=box["label"] 
        name=unicodedata.normalize('NFKD', name).encode('ascii','ignore')
        xmin= int(box["topleft"]["x"])
        ymin= int(box["topleft"]["y"]) 
        xmax= int(box["bottomright"]["x"])
        ymax= int(box["bottomright"]["y"])
        array=[xmin,ymin,xmax,ymax]
        lis.append(array)
        lis_name.append(name)
    
    return np.asarray(lis).reshape([-1,4]),lis_name
 
 
    

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	 
	interArea = (xB - xA + 1) * (yB - yA + 1)
	
	dx = (xB-xA+1)
	dy = (yB - yA + 1)
	if (dx>=0) and (dy>=0):
		interArea=dx*dy
	else:
		interArea=0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea =(boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	totalArea=float(boxAArea + boxBArea - interArea)
 	#print interArea , totalArea
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / totalArea
 
	# return the intersection over union value
	return iou
 
import os,glob
def find_accuracy(path,jpath):
    
    os.chdir(path)
    accuracies=[]
    for b,file in enumerate(glob.glob("*.xml")):
         
        
        groundtruth,names=get_box_from_xml(file)
         
         
        d_path=jpath+file.split(".")[0]+".json"
        detected,names_det=get_box_from_json(d_path) 
        false_positives=0;
         
        names=[unicodedata.normalize('NFKD', name).encode('ascii','ignore') for name in names]
         
        undetected=[]
        False_Positive=[]
        True_Positive=[]

        for i in range(0, len(groundtruth)):
            name=names[i]
            
            
            max=0.0; match=-1     
             
            if(name in names_det): 
                index = [b for b, x in enumerate(names_det) if x == name]
                
                for jj,j in enumerate(detected[index] ):
                     
                    IoU=bb_intersection_over_union(groundtruth[i] ,j)
                    #print IoU
                    if IoU>max:
                        max=IoU
                        match=j
                     
            if max<0.5:
                undetected.append(groundtruth[i])
            elif max>=0.5:
                True_Positive.append((groundtruth[i] ,match))

        
        falses=np.asarray(detected) 
        
        
       
        for box in falses:
            flag=0
            for pair in True_Positive:
                if np.equal(box,pair[1]).all():
                    flag=1  
                    #print "found box in the detected", box
            if flag==0:
                False_Positive.append(box)
        
         
        print ("undetected:", len(undetected), "    True positives:", len(True_Positive), "   False positives:", len(False_Positive))
        accuracy=float(2*len(True_Positive))/(2*len(True_Positive)+len(False_Positive)+len(undetected))
        accuracies.append(accuracy)
        
    print "Final Accuracy", np.mean(np.asarray(accuracies)) 
 
 


def main():
    annotation_path="/home/omer/Desktop/darkflow/test/training/annotations/"    
    json_path="/home/omer/Desktop/darkflow/test/training/json/"    
    find_accuracy(annotation_path,json_path)


main()
