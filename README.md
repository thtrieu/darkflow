# SORCIT Yolo 
 Original Darkflow Repo - https://github.com/thtrieu/darkflow
 
## Link for Downloading weights file
https://www.google.com/url?q=https%3A%2F%2Foc.codespring.ro%2Fs%2FJgyo6N4Jen3ma2P%2Fdownload&sa=D&sntz=1&usg=AFQjCNE24k8FoSABZqzLnh7f2yIw_M5FIA

# Yolo Helper
Annotation Tool Written in  Python for Quick Image Annotation. The scripts are located in `/annotation`.

## Installation
Clone this repository, create a new directory called `images` and add your images there. It will export xml annotation files to `annotations` folder which will be created by default if it doesn't exist.

## Run 

`python3 annotation.py`

## Annotating

This program will open every image and you can start drawing boxes surrounding objects that YOLO needs to recognize. 
If image contains more objects, once you finished drawing the first box, start drawing the second one. The first box will disappear, but the data will be saved. You can do this for an unlimited number of times. Once you annotated all objects - hit `q` and move to next image.
