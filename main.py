from darknet import *
from tfnet import *
from tensorflow import flags

flags.DEFINE_string("testset", "test", "path to testing directory")
flags.DEFINE_string("dataset", "../pascal/VOCdevkit/IMG/", "path to dataset directory")
flags.DEFINE_string("annotation", "../pascal/VOCdevkit/ANN/", "path to annotation directory")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "3c", "configuration of choice")
flags.DEFINE_boolean("train", False, "training mode or not?")
flags.DEFINE_integer("load", 0, "load a saved backup/checkpoint, -1 for newest")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "How much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "Learning rate")
flags.DEFINE_integer("keep",20,"Number of most recent training results to save")
flags.DEFINE_integer("batch", 12, "Batch size")
flags.DEFINE_integer("epoch", 1000, "Number of epoch")
flags.DEFINE_integer("save", 2000, "Save checkpoint every ? training examples")

FLAGS = flags.FLAGS
image = FLAGS.dataset
annot = FLAGS.annotation + 'parsed.bin'

step = int()
if FLAGS.load < 0:
	try:
		with open('backup/checkpoint','r') as f:
			lines = f.readlines()
	except:
		sys.exit('Seems like there is no recent training in backup/')
	name = lines[-1].split(' ')[1].split('"')[1]
	step = int(name.split('-')[1])
else: step = FLAGS.load
yoloNet = Darknet(FLAGS.model + int(step > 0) * '-{}'.format(step))

print ('\nCompiling net & fill in parameters...')
start = time.time()
if FLAGS.gpu <= 0.:
	with tf.device('cpu:0'):
		tfnet = TFNet(yoloNet, FLAGS)
else:
	tfnet = TFNet(yoloNet, FLAGS)
tfnet.step = step
tfnet.setup_meta_ops()
print ('Finished in {}s'.format(time.time() - start))

if FLAGS.train:
	print '\nEnter training ...'
	tfnet.train(image, annot, FLAGS.batch, FLAGS.epoch)

print
tfnet.predict()