"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os

old_graph_msg = 'Resolving old graph def {} (no guarantee)'

def build_train_op(self): 
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    gradients = optimizer.compute_gradients(self.framework.loss)
    self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
    if self.FLAGS.load < 0: # load lastest ckpt
        with open(os.path.join(self.FLAGS.backup, 'checkpoint'), 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)
    
    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try: self.saver.restore(self.sess, load_point)
    except: load_old_graph(self, load_point)

def say(self, *msgs): # Verbose, prints provided messages while building CG
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)

def load_old_graph(self, ckpt): 
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))
    
    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
        'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})

def _get_fps(self, frame): #check for FPS
    elapsed = int()  
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame, False)
    return timer() - start

def camera(self): # Only for demo run, takes a video from camera or a given file, '0 stands for camera, '
    file = self.FLAGS.demo # input file given, 0 for webcam feed
    SaveVideo = self.FLAGS.saveVideo # bool
    
    if file == 'camera': 
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)
        
    camera = cv2.VideoCapture(file) # Capture video from said file and save it to variable
    
    if file == 0: # On line Verbose
        self.say('Press [ESC] to quit demo')
        
    assert camera.isOpened(), \
    'Cannot capture source' # Check if camera is available
    
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read() # reading useful input from camera
        height, width, _ = frame.shape # getting raw input shape
        cv2.resizeWindow('', width, height) # resize window equal to width and height of input
    else:
        _, frame = camera.read() # If video just take the height and width
        height, width, _ = frame.shape

    if SaveVideo: # If True
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Write to file
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame) # _get_fps gives time per frame, inverse gives real FPS
          if fps < 1:
            fps = 1 # if the fps is lower than 1 return 1 else get the actual Proper FPS after rounding it up
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            'video.avi', fourcc, fps, (width, height)) #Actual video write at a given fps

    # buffers for demo in batch
    buffer_inp = list() # Input stream into a list/ buffer
    buffer_pre = list() # Preprocessed stream into save/ buffer
    
    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened(): #While the camera is opened
        elapsed += 1
        _, frame = camera.read() # read the frame
        if frame is None: # frame will be None at the end of video
            print ('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame) #Preprocess pipeline
        buffer_inp.append(frame) # append input frames
        buffer_pre.append(preprocessed) # append preprocessed frame
        
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0: # demo can take multiple videos in queue
            feed_dict = {self.inp: buffer_pre} # If all the buffer is loaded feed the preprocessed buffer as input to CG
            net_out = self.sess.run(self.out, feed_dict) # Run a session on CG
            for img, single_out in zip(buffer_inp, net_out): # Clubbed for loop
                postprocessed = self.framework.postprocess( # postprocess input buffer using model output per frame
                    single_out, img, False)
                if SaveVideo: # if True, save Video
                    videoWriter.write(postprocessed)
                if file == 0: #camera window
                    cv2.imshow('', postprocessed) # If camera feed, just show the output as a feed
            # Clear Buffers
            buffer_inp = list() # to save memory
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush() # at intervals STDOUT the average frame rate
        if file == 0: #camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release() # Release video capturing
    if file == 0: #camera window
        cv2.destroyAllWindows() 

def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt
