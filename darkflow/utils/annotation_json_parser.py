#json format parser
# author: chadrick.kwag@gmail.com

# most part of the code is just copied from pascal_voc_clean_xml.py

import json
import os
import sys
import glob



def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))

def chadrick_jsonparser(ANN, pick, exclusive = False):
    # ANN = FLAGS.annotation -> annotation dir
    # pick = meta['labels']
    print("using chadrick json parser")
    print('using chadrick')

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

        # the file containds zero padding and the actual json is in the first line
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

                
                # xn = int(float(xmlbox.find('xmin').text))
                # xx = int(float(xmlbox.find('xmax').text))
                # yn = int(float(xmlbox.find('ymin').text))
                # yx = int(float(xmlbox.find('ymax').text))

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