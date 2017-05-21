from .defaults import argHandler #Import the default arguments
import os

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    from darkflow.net.build import TFNet

    # make sure all necessary dirs exist
    def get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)

    if FLAGS.profile:
        tfnet.framework.profile(tfnet)
        exit()

    if FLAGS.demo:
        tfnet.camera(FLAGS.demo, FLAGS.saveVideo)
        exit()

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: exit('Training finished')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()