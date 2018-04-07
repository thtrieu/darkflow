from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet
import time


def cliHandler(args):
    start_time = time.time()
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)

    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)

    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb:
            print("Finished in %s seconds " % int(time.time() - start_time))
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print("Finished in %s seconds " % int(time.time() - start_time))
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()
