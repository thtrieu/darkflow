"""
json format parser
author: chadrick.kwag@gmail.com

most part of the code is just copied from pascal_voc_clean_xml.py

the format of the json file should be like the following example:

{"imgfile": "0313.png", "w": 640, "h": 480, "objects": [{"rect": {"y1": 4, "y2": 144, "x1": 385, "x2": 587}, "name": "face"}]}

the json file should be in a single line.
it is convenient to use the python's json module when creating these files.

also, this parser checks the size comparison of x1/x2 and y1/y2.
when this size rule is broken and mapped to xn,yn,xx,yx, then it will cause and error during training.

"""

import json
import os
import sys
import glob



def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))

def annotation_json_parser(ANN, pick, exclusive = False):
    # ANN = FLAGS.annotation -> annotation dir
    # pick = meta['labels']

    dumps= list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.json')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar      
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()
        
        # actual parsing 
        print("opening file {}".format(file))
        in_file = open(file)

        # the file contains zero padding and the actual json is in the first line
        firstline = in_file.readline()
        firstline = firstline.replace('\0','')

        root = json.loads(firstline)


        imgfile = str(root['imgfile'])
        
        w = root['w']
        h = root['h']
        all = list()
        objects = root['objects']
        for obj in objects:
                current = list()
                name = str(obj['name'])
                if name not in pick:
                    print("{} not in pick".format(name))
                    continue

                rect = obj['rect']

                # xn = x1, xx = x2, yn = y1, yx = y2

                xn = rect['x1']
                xx = rect['x2']
                yn = rect['y1']
                yx = rect['y2']

                # safety check for min/max

                if xn>xx:
                    xx = rect['x1']
                    xn = rect['x2']

                if yn > yx :
                    yn = rect['y2']
                    yx = rect['y1']


                current = [name,xn,yn,xx,yx]
                all += [current]

        add = [[imgfile, [w, h, all]]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps