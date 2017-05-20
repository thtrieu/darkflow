from darkflow.net.build import TFNet
import json
import requests
import cv2
import os
import sys
import pytest

#NOTE: This file is designed to be run in the TravisCI environment. If you want to run it locally set the environment variable TRAVIS_BUILD_DIR to the base
#      directory of the cloned darkflow repository. WARNING: This file will make some modifications to the darkflow repository during testing such as
#      deleting images from sample_img/ that won't be used for testing and renaming flow to flow.py (so don't run it locally if you don't want this happening!)

#Settings
imgWidth = 640
imgHeight = 424
buildPath = os.environ.get("TRAVIS_BUILD_DIR")
if buildPath is None:
    buildPath = os.environ.get("WERCKER_ROOT")
if os.name == 'nt':
    buildPath = r"C:\Users\abags\OneDrive\School\Computer_Programming\salieo\darkflow" #We're running locally on Windows (dev hopefully!) #REMOVE THIS
if buildPath is None:
    print()
    print("TRAVIS_BUILD_DIR environment variable was not found - is this running on TravisCI?")
    print("If you want to test this locally, set TRAVIS_BUILD_DIR to the base directory of the cloned darkflow repository.")
    exit()
testImgPath = os.path.join(buildPath, "sample_img", "sample_person.jpg")
expectedDetectedObjectsV1 = [{"label": "dog","confidence": 0.46,"topleft": {"x": 84, "y": 249},"bottomright": {"x": 208,"y": 367}}, 
                             {"label": "person","confidence": 0.60,"topleft": {"x": 159, "y": 102},"bottomright": {"x": 304,"y": 365}}]

expectedDetectedObjectsV2 = [{"label":"person","confidence":0.82,"topleft":{"x":189,"y":96},"bottomright":{"x":271,"y":380}},
                           {"label":"dog","confidence":0.79,"topleft":{"x":69,"y":258},"bottomright":{"x":209,"y":354}},
                           {"label":"horse","confidence":0.89,"topleft":{"x":397,"y":127},"bottomright":{"x":605,"y":352}}]
posCompareThreshold = 0.05 #Comparisons must match be within 5% of width/height when compared to expected value
threshCompareThreshold = 0.1 #Comparisons must match within 0.1 of expected threshold for each prediction
yoloDownloadV1 = "https://pjreddie.com/media/files/yolo-small.weights"
yoloDownloadV2 = "https://pjreddie.com/media/files/yolo.weights"

def download_file(url, savePath):
    fileName = savePath.split("/")[-1]
    if not os.path.isfile(savePath):
        os.makedirs(os.path.dirname(savePath), exist_ok=True) #Make directories nessecary for file incase they don't exist
        print("Downloading " + fileName + " file...")
        r = requests.get(url, stream=True)
        with open(savePath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        r.close()
    else:
        print("Found existing " + fileName + " file.")

if os.path.isfile(os.path.join(buildPath, "flow")):
    os.rename(os.path.join(buildPath, "flow"), os.path.join(buildPath, "flow.py")) #Change flow to flow.py so we can import it
sys.path.insert(0, buildPath) #Add the buildPath to the PATH
from flow import main

yoloWeightPathV1 = os.path.join(buildPath, "bin", yoloDownloadV1.split("/")[-1])
yoloCfgPathV1 = os.path.join(buildPath, "cfg", "v1", "{0}.cfg".format(os.path.splitext(os.path.basename(yoloWeightPathV1))[0]))

yoloWeightPathV2 = os.path.join(buildPath, "bin", yoloDownloadV2.split("/")[-1])
yoloCfgPathV2 = os.path.join(buildPath, "cfg", "{0}.cfg".format(os.path.splitext(os.path.basename(yoloWeightPathV2))[0]))

pbPath = os.path.join(buildPath, "built_graph", os.path.splitext(os.path.basename(yoloWeightPathV2))[0] + ".pb")
metaPath = os.path.join(buildPath, "built_graph", os.path.splitext(os.path.basename(yoloWeightPathV2))[0] + ".meta")

generalConfigPath = os.path.join(buildPath, "cfg")

download_file(yoloDownloadV1, yoloWeightPathV1) #Check if we need to download (and if so download) the YOLOv1 weights
download_file(yoloDownloadV2, yoloWeightPathV2) #Check if we need to download (and if so download) the YOLOv2 weights

def executeCLI(commandString):
    print()
    print("Executing: {0}".format(commandString))
    print()
    splitArgs = [item.strip() for item in commandString.split(" ")]
    main(splitArgs) #Run the command
    print()

def compareSingleObjects(firstObject, secondObject, width, height):
    if(abs(firstObject["topleft"]["x"] - secondObject["topleft"]["x"]) > width * posCompareThreshold):
        return False
    if(abs(firstObject["topleft"]["y"] - secondObject["topleft"]["y"]) > height * posCompareThreshold):
        return False
    if(abs(firstObject["bottomright"]["x"] - secondObject["bottomright"]["x"]) > width * posCompareThreshold):
        return False
    if(abs(firstObject["bottomright"]["y"] - secondObject["bottomright"]["y"]) > height * posCompareThreshold):
        return False
    if(abs(firstObject["confidence"] - secondObject["confidence"]) > threshCompareThreshold):
        return False
    return True

def compareObjectData(defaultObjects, newObjects, width, height):
    currentlyFound = False
    for firstObject in defaultObjects:
        currentlyFound = False
        for secondObject in newObjects:
            if compareSingleObjects(firstObject, secondObject, width, height):
                currentlyFound = True
                break
        if not currentlyFound:
            return False
    return True

#Delete all images that won't be tested on so forwarding the whole folder doesn't take forever
filelist = [f for f in os.listdir(os.path.dirname(testImgPath)) if os.path.isfile(os.path.join(os.path.dirname(testImgPath), f)) and f != os.path.basename(testImgPath)]
for f in filelist:
    os.remove(os.path.join(os.path.dirname(testImgPath), f))

def test_CLI_IMG_YOLOv2():
    #Test predictions outputted to an image using the YOLOv2 model through CLI
    #NOTE: This test currently does not verify anything about the image created (i.e. proper labeling, proper positioning of prediction boxes, etc.)
    #      it simply verifies that the code executes properly and that the expected output image is indeed created in ./test/img/out

    testString = "./flow --imgdir {0} --model {1} --load {2} --config {3} --threshold 0.4".format(os.path.dirname(testImgPath), yoloCfgPathV2, yoloWeightPathV2, generalConfigPath)
    executeCLI(testString)

    outputImgPath = os.path.join(os.path.dirname(testImgPath), "out", os.path.basename(testImgPath))
    assert os.path.exists(outputImgPath), "Expected output image: {0} was not found.".format(outputImgPath)

def test_CLI_JSON_YOLOv2():
    #Test predictions outputted to a JSON file using the YOLOv2 model through CLI
    #NOTE: This test verifies that the code executes properly, the JSON file is created properly and the predictions generated are within a certain
    #      margin of error when compared to the expected predictions.

    testString = "./flow --imgdir {0} --model {1} --load {2} --config {3} --threshold 0.4 --json".format(os.path.dirname(testImgPath), yoloCfgPathV2, yoloWeightPathV2, generalConfigPath)
    executeCLI(testString)

    outputJSONPath = os.path.join(os.path.dirname(testImgPath), "out", os.path.splitext(os.path.basename(testImgPath))[0] + ".json")
    assert os.path.exists(outputJSONPath), "Expected output JSON file: {0} was not found.".format(outputJSONPath)

    with open(outputJSONPath) as json_file:
        loadedPredictions = json.load(json_file)

    assert compareObjectData(expectedDetectedObjectsV2, loadedPredictions, imgWidth, imgHeight), "Generated object predictions from JSON were not within margin of error compared to expected values."

def test_CLI_SAVEPB_YOLOv2():
    #Save .pb and .meta as generated from the YOLOv2 model through CLI
    #NOTE: This test verifies that the code executes properly, and the .pb and .meta files are successfully created. A subsequent test will verify the
    #      contents of those files.

    testString = "./flow --model {0} --load {1} --config {2} --threshold 0.4 --savepb".format(yoloCfgPathV2, yoloWeightPathV2, generalConfigPath)
    
    with pytest.raises(SystemExit):
            executeCLI(testString)

    assert os.path.exists(pbPath), "Expected output .pb file: {0} was not found.".format(pbPath)
    assert os.path.exists(metaPath), "Expected output .meta file: {0} was not found.".format(metaPath)

def test_RETURNPREDICT_PBLOAD_YOLOv2():
    #Test the .pb and .meta files generated in the previous step
    #NOTE: This test verifies that the code executes properly, and the .pb and .meta files that were created are able to be loaded and used for inference.
    #      The predictions that are generated will be compared against expected predictions.

    options = {"pbLoad": pbPath, "metaLoad": metaPath, "threshold": 0.4}
    tfnet = TFNet(options)
    imgcv = cv2.imread(testImgPath)
    loadedPredictions = tfnet.return_predict(imgcv)

    assert compareObjectData(expectedDetectedObjectsV2, loadedPredictions, imgWidth, imgHeight), "Generated object predictions from return_predict() were not within margin of error compared to expected values."

def test_RETURNPREDICT_YOLOv1():
    #Test YOLOv1 using normal .weights and .cfg
    #NOTE: This test verifies that the code executes properly, and that the predictions generated are within the accepted margin of error to the expected predictions.

    options = {"model": yoloCfgPathV1, "load": yoloWeightPathV1, "config": generalConfigPath, "threshold": 0.4}
    tfnet = TFNet(options)
    imgcv = cv2.imread(testImgPath)
    loadedPredictions = tfnet.return_predict(imgcv)

    assert compareObjectData(expectedDetectedObjectsV2, loadedPredictions, imgWidth, imgHeight), "Generated object predictions from return_predict() were not within margin of error compared to expected values."