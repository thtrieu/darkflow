from darkflow.net.build import TFNet
from darkflow.cli import cliHandler
import json
import requests
import cv2
import os
import sys
import pytest

#NOTE: This file is designed to be run in the TravisCI environment. If you want to run it locally set the environment variable TRAVIS_BUILD_DIR to the base
#      directory of the cloned darkflow repository. WARNING: This file delete images from sample_img/ that won't be used for testing (so don't run it
#      locally if you don't want this happening!)

#Settings
buildPath = os.environ.get("TRAVIS_BUILD_DIR")

if buildPath is None:
    print()
    print("TRAVIS_BUILD_DIR environment variable was not found - is this running on TravisCI?")
    print("If you want to test this locally, set TRAVIS_BUILD_DIR to the base directory of the cloned darkflow repository.")
    exit()

testImg = {"path": os.path.join(buildPath, "sample_img", "sample_person.jpg"), "width": 640, "height": 424,
           "expected-objects": {"yolo-small": [{"label": "dog", "confidence": 0.46, "topleft": {"x": 84, "y": 249}, "bottomright": {"x": 208, "y": 367}},
                                                  {"label": "person", "confidence": 0.60, "topleft": {"x": 159, "y": 102}, "bottomright": {"x": 304, "y": 365}}],
                                "yolo":       [{"label": "person", "confidence": 0.82, "topleft": {"x": 189, "y": 96}, "bottomright": {"x": 271, "y": 380}},
                                                  {"label": "dog", "confidence": 0.79, "topleft": {"x": 69, "y": 258}, "bottomright": {"x": 209, "y": 354}},
                                                  {"label": "horse", "confidence": 0.89, "topleft": {"x": 397, "y": 127}, "bottomright": {"x": 605, "y": 352}}]}}

trainImgBikePerson = {"path": os.path.join(buildPath, "test", "training", "images", "1.jpg"), "width": 500, "height": 375,
                      "expected-objects": {"tiny-yolo-voc": [{"label":"bicycle","confidence":0.35,"topleft":{"x":121,"y":126},"bottomright":{"x":233,"y":244}},
                                                             {"label":"person","confidence":0.60,"topleft":{"x":132,"y":35},"bottomright":{"x":232,"y":165}}]}}

trainImgHorsePerson = {"path": os.path.join(buildPath, "test", "training", "images", "2.jpg"), "width": 500, "height": 332,
                       "expected-objects": {"tiny-yolo-voc": [{"label":"horse","confidence":0.99,"topleft":{"x":156,"y":108},"bottomright":{"x":410,"y":281}},
                                                              {"label":"person","confidence":0.89,"topleft":{"x":258,"y":52},"bottomright":{"x":300,"y":218}}]}}


posCompareThreshold = 0.05 #Comparisons must match be within 5% of width/height when compared to expected value
threshCompareThreshold = 0.1 #Comparisons must match within 0.1 of expected threshold for each prediction

yolo_small_Download = "https://pjreddie.com/media/files/yolo-small.weights" #YOLOv1
yolo_Download = "https://pjreddie.com/media/files/yolo.weights" #YOLOv2
tiny_yolo_voc_Download = "https://pjreddie.com/media/files/tiny-yolo-voc.weights" #YOLOv2

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

yolo_small_WeightPath = os.path.join(buildPath, "bin", yolo_small_Download.split("/")[-1])
yolo_small_CfgPath = os.path.join(buildPath, "cfg", "v1", "{0}.cfg".format(os.path.splitext(os.path.basename(yolo_small_WeightPath))[0]))

yolo_WeightPath = os.path.join(buildPath, "bin", yolo_Download.split("/")[-1])
yolo_CfgPath = os.path.join(buildPath, "cfg", "{0}.cfg".format(os.path.splitext(os.path.basename(yolo_WeightPath))[0]))

tiny_yolo_voc_WeightPath = os.path.join(buildPath, "bin", tiny_yolo_voc_Download.split("/")[-1])
tiny_yolo_voc_CfgPath = os.path.join(buildPath, "cfg", "{0}.cfg".format(os.path.splitext(os.path.basename(tiny_yolo_voc_WeightPath))[0]))

pbPath = os.path.join(buildPath, "built_graph", os.path.splitext(os.path.basename(yolo_WeightPath))[0] + ".pb")
metaPath = os.path.join(buildPath, "built_graph", os.path.splitext(os.path.basename(yolo_WeightPath))[0] + ".meta")

generalConfigPath = os.path.join(buildPath, "cfg")

download_file(yolo_small_Download, yolo_small_WeightPath) #Check if we need to download (and if so download) the yolo-small weights (YOLOv1)
download_file(yolo_Download, yolo_WeightPath) #Check if we need to download (and if so download) the yolo weights (YOLOv2)
download_file(tiny_yolo_voc_Download, tiny_yolo_voc_WeightPath) #Check if we need to download (and if so download) the tiny-yolo-voc weights (YOLOv2)

def executeCLI(commandString):
    print()
    print("Executing: {0}".format(commandString))
    print()
    splitArgs = [item.strip() for item in commandString.split(" ")]
    cliHandler(splitArgs) #Run the command
    print()

def compareSingleObjects(firstObject, secondObject, width, height, threshCompare, posCompare):
    if(firstObject["label"] != secondObject["label"]):
        return False
    if(abs(firstObject["topleft"]["x"] - secondObject["topleft"]["x"]) > width * posCompare):
        return False
    if(abs(firstObject["topleft"]["y"] - secondObject["topleft"]["y"]) > height * posCompare):
        return False
    if(abs(firstObject["bottomright"]["x"] - secondObject["bottomright"]["x"]) > width * posCompare):
        return False
    if(abs(firstObject["bottomright"]["y"] - secondObject["bottomright"]["y"]) > height * posCompare):
        return False
    if(abs(firstObject["confidence"] - secondObject["confidence"]) > threshCompare):
        return False
    return True

def compareObjectData(defaultObjects, newObjects, width, height, threshCompare, posCompare):
    currentlyFound = False
    for firstObject in defaultObjects:
        currentlyFound = False
        for secondObject in newObjects:
            if compareSingleObjects(firstObject, secondObject, width, height, threshCompare, posCompare):
                currentlyFound = True
                break
        if not currentlyFound:
            return False
    return True

#Delete all images that won't be tested on so forwarding the whole folder doesn't take forever
filelist = [f for f in os.listdir(os.path.dirname(testImg["path"])) if os.path.isfile(os.path.join(os.path.dirname(testImg["path"]), f)) and f != os.path.basename(testImg["path"])]
for f in filelist:
    os.remove(os.path.join(os.path.dirname(testImg["path"]), f))


#TESTS FOR INFERENCE
def test_CLI_IMG_YOLOv2():
    #Test predictions outputted to an image using the YOLOv2 model through CLI
    #NOTE: This test currently does not verify anything about the image created (i.e. proper labeling, proper positioning of prediction boxes, etc.)
    #      it simply verifies that the code executes properly and that the expected output image is indeed created in ./test/img/out

    testString = "flow --imgdir {0} --model {1} --load {2} --config {3} --threshold 0.4".format(os.path.dirname(testImg["path"]), yolo_CfgPath, yolo_WeightPath, generalConfigPath)
    executeCLI(testString)

    outputImgPath = os.path.join(os.path.dirname(testImg["path"]), "out", os.path.basename(testImg["path"]))
    assert os.path.exists(outputImgPath), "Expected output image: {0} was not found.".format(outputImgPath)
    os.remove(outputImgPath) #Remove the image so that it does not affect subsequent tests

def test_CLI_JSON_YOLOv2():
    #Test predictions outputted to a JSON file using the YOLOv2 model through CLI
    #NOTE: This test verifies that the code executes properly, the JSON file is created properly and the predictions generated are within a certain
    #      margin of error when compared to the expected predictions.

    testString = "flow --imgdir {0} --model {1} --load {2} --config {3} --threshold 0.4 --json".format(os.path.dirname(testImg["path"]), yolo_CfgPath, yolo_WeightPath, generalConfigPath)
    executeCLI(testString)

    outputJSONPath = os.path.join(os.path.dirname(testImg["path"]), "out", os.path.splitext(os.path.basename(testImg["path"]))[0] + ".json")
    assert os.path.exists(outputJSONPath), "Expected output JSON file: {0} was not found.".format(outputJSONPath)

    with open(outputJSONPath) as json_file:
        loadedPredictions = json.load(json_file)

    assert compareObjectData(testImg["expected-objects"]["yolo"], loadedPredictions, testImg["width"], testImg["height"], threshCompareThreshold, posCompareThreshold), "Generated object predictions from JSON were not within margin of error compared to expected values."
    os.remove(outputJSONPath) #Remove the JSON file so that it does not affect subsequent tests

def test_CLI_JSON_YOLOv1():
    #Test predictions outputted to a JSON file using the YOLOv1 model through CLI
    #NOTE: This test verifies that the code executes properly, the JSON file is created properly and the predictions generated are within a certain
    #      margin of error when compared to the expected predictions.

    testString = "flow --imgdir {0} --model {1} --load {2} --config {3} --threshold 0.4 --json".format(os.path.dirname(testImg["path"]), yolo_small_CfgPath, yolo_small_WeightPath, generalConfigPath)
    executeCLI(testString)

    outputJSONPath = os.path.join(os.path.dirname(testImg["path"]), "out", os.path.splitext(os.path.basename(testImg["path"]))[0] + ".json")
    assert os.path.exists(outputJSONPath), "Expected output JSON file: {0} was not found.".format(outputJSONPath)

    with open(outputJSONPath) as json_file:
        loadedPredictions = json.load(json_file)

    assert compareObjectData(testImg["expected-objects"]["yolo-small"], loadedPredictions, testImg["width"], testImg["height"], threshCompareThreshold, posCompareThreshold), "Generated object predictions from JSON were not within margin of error compared to expected values."
    os.remove(outputJSONPath) #Remove the JSON file so that it does not affect subsequent tests

def test_CLI_SAVEPB_YOLOv2():
    #Save .pb and .meta as generated from the YOLOv2 model through CLI
    #NOTE: This test verifies that the code executes properly, and the .pb and .meta files are successfully created. The subsequent test will verify the
    #      contents of those files.

    testString = "flow --model {0} --load {1} --config {2} --threshold 0.4 --savepb".format(yolo_CfgPath, yolo_WeightPath, generalConfigPath)
    
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
    imgcv = cv2.imread(testImg["path"])
    loadedPredictions = tfnet.return_predict(imgcv)

    assert compareObjectData(testImg["expected-objects"]["yolo"], loadedPredictions, testImg["width"], testImg["height"], threshCompareThreshold, posCompareThreshold), "Generated object predictions from return_predict() were not within margin of error compared to expected values."

#TESTS FOR TRAINING
def test_TRAIN_FROM_WEIGHTS_CLI__LOAD_CHECKPOINT_RETURNPREDICT_YOLOv2():
    #Test training using pre-generated weights for tiny-yolo-voc
    #NOTE: This test verifies that the code executes properly, and that the expected checkpoint file (tiny-yolo-voc-20.meta in this case) is generated.
    #      In addition, predictions are generated using the checkpoint file to verify that training completed successfully.

    testString = "flow --model {0} --load {1} --train --dataset {2} --annotation {3} --epoch 20".format(tiny_yolo_voc_CfgPath, tiny_yolo_voc_WeightPath, os.path.join(buildPath, "test", "training", "images"), os.path.join(buildPath, "test", "training", "annotations"))
    with pytest.raises(SystemExit):
        executeCLI(testString)

    checkpointPath = os.path.join(buildPath, "ckpt", "tiny-yolo-voc-20.meta")
    assert os.path.exists(checkpointPath), "Expected output checkpoint file: {0} was not found.".format(checkpointPath)

    #Using trained weights
    options = {"model": tiny_yolo_voc_CfgPath, "load": 20, "config": generalConfigPath, "threshold": 0.1}
    tfnet = TFNet(options)

    #Make sure predictions very roughly match the expected values for image with bike and person
    imgcv = cv2.imread(trainImgBikePerson["path"])
    loadedPredictions = tfnet.return_predict(imgcv)
    assert compareObjectData(trainImgBikePerson["expected-objects"]["tiny-yolo-voc"], loadedPredictions, trainImgBikePerson["width"], trainImgBikePerson["height"], 0.7, 0.25), "Generated object predictions from training (for image with person on the bike) were not anywhere close to what they are expected to be.\nTraining may not have completed successfully."
    differentThanExpectedBike = compareObjectData(trainImgBikePerson["expected-objects"]["tiny-yolo-voc"], loadedPredictions, trainImgBikePerson["width"], trainImgBikePerson["height"], 0.01, 0.001)

    #Make sure predictions very roughly match the expected values for image with horse and person
    imgcv = cv2.imread(trainImgHorsePerson["path"])
    loadedPredictions = tfnet.return_predict(imgcv)
    assert compareObjectData(trainImgHorsePerson["expected-objects"]["tiny-yolo-voc"], loadedPredictions, trainImgHorsePerson["width"], trainImgHorsePerson["height"], 0.7, 0.25), "Generated object predictions from training (for image with person on the horse) were not anywhere close to what they are expected to be.\nTraining may not have completed successfully."
    differentThanExpectedHorse = compareObjectData(trainImgHorsePerson["expected-objects"]["tiny-yolo-voc"], loadedPredictions, trainImgHorsePerson["width"], trainImgHorsePerson["height"], 0.01, 0.001)

    assert not (differentThanExpectedBike and differentThanExpectedHorse), "The generated object predictions for both images appear to be exactly the same as the ones generated with the original weights.\nTraining may not have completed successfully.\n\nNOTE: It is possible this is a fluke error and training did complete properly (try running this build again to confirm) - but most likely something is wrong."