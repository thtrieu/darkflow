from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet


"""sets the FLAG dict used everywhere, checks/creates needed directories, call to build computation graph using FLAGS, train/demo/save/predict """
def cliHandler(args):
    FLAGS = argHandler() #FLAGS defined
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            # checking in a list of dirs, create one if not exists
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    #dirs from FLAGS
    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load) #load weights if default given else pass
    except: pass

    tfnet = TFNet(FLAGS) # Build TF graph with the given FLAGS
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()
