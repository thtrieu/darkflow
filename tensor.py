from Yolo import *
from box import *
from TFnet import *
from tensorflow import flags
import sys
import time
import os

flags.DEFINE_string("test", "data", "path to testing folder")
flags.DEFINE_string("pascal", "../pascal/VOCdevkit", "path to training set")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "3c", "yolo configuration of choice")
flags.DEFINE_boolean("train", False, "training mode or not?")
flags.DEFINE_boolean("load", False, "load the newest train in backup/checkpoint")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "How much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "Learning rate")
flags.DEFINE_string("scale", "1,1,.5,5.", 
	"Comma-separated scaling for probability, confidence, noobj, coordinate terms in the loss")
flags.DEFINE_integer("keep",20,"Number of most recent training results to save")
flags.DEFINE_integer("batch", 12, "Batch size")
flags.DEFINE_integer("epoch", 1000, "Number of epoch")
flags.DEFINE_integer("save", 2000, "Save checkpoint every ? training examples")
FLAGS = flags.FLAGS
image = FLAGS.pascal + '/IMG/'
annot = FLAGS.pascal + '/ANN/' + 'parsed.yolotf'

step = int()
if FLAGS.load:
	try:
		with open('backup/checkpoint','r') as f:
			lines = f.readlines()
	except:
		sys.exit('Seems like there is no recent training in backup/')
	name = lines[-1].split(' ')[1].split('"')[1]
	step = int(name.split('-')[1])
yoloNet = YOLO(FLAGS.model + int(step > 0) * '-{}'.format(step))

print ('Compiling net & initialise parameters...')
start = time.time()
if FLAGS.gpu <= 0.:
	with tf.device('cpu:0'):
		model = SimpleNet(yoloNet, FLAGS)
else:
	model = SimpleNet(yoloNet, FLAGS)
model.step = step
model.setup_meta_ops(FLAGS)
print ('Finished in {}s'.format(time.time() - start))

if FLAGS.train:
	print 'training mode'
	model.train(image, annot, FLAGS.batch, FLAGS.epoch)
model.predict(FLAGS)