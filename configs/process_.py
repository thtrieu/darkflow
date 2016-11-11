import numpy as np
import os
import sys

model = sys.argv[1]
undiscovered = True

# Step 1: parsing cfg file
with open('yolo-{}.cfg'.format(model), 'rb') as f:
    lines = f.readlines()

s = [] # contains layers' info
S = int() # the number of grid cell
add = dict()
for line in lines:
    line = line.strip()
    if 'side' in line:
        S = int(line.split('=')[1].strip())
    if '[' in line:
        if add != {}:
            s += [add]
        add = dict()
    else:
        try:
            i = float(line.split('=')[1].strip())
            if i == int(i): i = int(i)
            add[line.split('=')[0]] = i
        except:
            try:
                if line.split('=')[1] == 'leaky':
                    add[line.split('=')[0]] = 'leaky'
            except:
                pass

# Step 2: investigate the weight file
weightf = '../yolo-{}.weights'.format(model)
if undiscovered:
    allbytes = os.path.getsize(weightf.format(model))
    allbytes /= 4 # each float is 4 byte
    allbytes -= 4 # the first 4 bytes are darknet specifications
    last_convo = int() 
    for i, d in enumerate(s):
        if len(d) == 4:
            last_convo = i # the index of last convolution layer
    flag = False
    channel = 3 # initial number of channel in the tensor volume
    out = int() 
    for i, d in enumerate(s):
        if 'batch' in d: continue
        if 'crop_width' in d: continue
        if 'side' in d: continue
        # for each iteration in this loop
        # allbytes will be gradually subtracted
        # by the size of the corresponding layer (d)
        # except for the 1st dense layer
        # it should be what remains after subtracting
        # all other layers
        if len(d) >= 4:
            allbytes -= d['size'] ** 2 * channel * d['filters']
            allbytes -= d['filters']
            channel = d['filters']
            if 'batch_normalize' in d:
                allbytes -= 2 * d['filters']
        elif 'output' in d: # this is a dense layer
            if flag is False: # this is the first dense layer
                out = out1 = d['output'] # output unit of the 1st dense layer
                flag = True # mark that the 1st dense layer is passed
                continue # don't do anything with the 1st dense layer
            allbytes -= out * d['output']
            allbytes -= d['output']
            out = d['output']
    allbytes -= out1 # substract the bias
    if allbytes <= 0:
            message = "Error: yolo-{}.cfg suggests a bigger size"
            message += " than yolo-{}.weights actually is"
            print message.format(model, model)
            assert allbytes > 0
    # allbytes is now = I * out1
    # where I is the input size of the 1st dense layer
    # I is also the volume of the last convolution layer
    # I = size * size * channel
    size = (np.sqrt(allbytes/out1/channel)) 
    print size
    size = int(size)
    n = last_convo + 1
    while 'output' not in s[n]:
        size *= s[n].get('size',1)
        n += 1
else:
    last_convo = None
    size = None

# Step 3: printing config
w = 448
h = 448
c = 3
l = w * h * c
flat = False

for i, d in enumerate(s):
    if 'batch' in d: continue
    if 'crop_width' in d: continue
    if 'side' in d: continue

    flag = False # flag for passing the 1st dense layer
    if len(d) >= 4:
        mult = (d['size'] == 3) 
        mult *= (d['stride'] != 2) + 1.
        if d['size'] == 1: d['pad'] = 0
        new = (w + mult * d['pad'] - d['size'])
        new /= d['stride']
        new = int(np.floor(new + 1.))
        if i == last_convo:
            # yield the negative expected size
            # instead of the indicated pad.
            d['pad'] = -size 
            new = size
        batch_norm = d.get('batch_normalize', 0)
        print ['conv', d['size'], c, d['filters'], 
            h, w, d['stride'], d['pad'], batch_norm]
        w = h = new
        c = d['filters']
        l = w * h * c
        if 'activation' in d:
            print ['LEAKY']
    if len(d) == 2:
        if 'output' not in d:
            print ['pool', d['size'], 0, 
                0, 0, 0, d['stride'], 0]
            new = (w     * 1.0 - d['size'])/d['stride'] + 1
            new = int(np.floor(new))
            w = h = new
            l = w * h * c
        else:
            if not flat:
                flat = True
                print ['FLATTEN']
            print ['conn', 0, 0,
            0, 0, 0, l, d['output']]
            l = d['output']
            if 'activation' in d:
                print ['LEAKY']
    if len(d) == 1:
        if 'output' not in d:
            print ['DROPOUT']
        else:
            if not flat:
                flat = True
                print ['FLATTEN']
            print ['conn', 0, 0,
            0, 0, 0, l, d['output']]
            l = d['output']