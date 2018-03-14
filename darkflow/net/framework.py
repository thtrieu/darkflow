from . import yolo
from . import yolov2
from . import vanilla
from os.path import basename


#OOPed 
#frameworks are made as classes, desired framework is initialized

class framework(object):
    constructor = vanilla.constructor #FCF

# frameworks are made as classes, desired framework is initialized. Instances will be made for train, val and test seperately


class framework(object):
    constructor = vanilla.constructor  # FCF

    loss = vanilla.train.loss

    def __init__(self, meta, FLAGS):
        model = basename(meta['model'])

        model = '.'.join(model.split('.')[:-1])#removing the last element from the model name
        meta['name'] = model 
        
        self.constructor(meta, FLAGS) #passing it to vanilla constructor that essentially does nothing

        model = '.'.join(model.split('.')[:-1])  # removing the last element from the model name
        meta['name'] = model

        self.constructor(meta, FLAGS)  # passing it to vanilla constructor that essentially does nothing


    def is_inp(self, file_name):
        return True


class YOLO(framework): #both YOLO and v2 are childs of framework class


class YOLO(framework):

    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolo.data.shuffle
    preprocess = yolo.predict.preprocess
    postprocess = yolo.predict.postprocess
    loss = yolo.train.loss
    is_inp = yolo.misc.is_inp
    profile = yolo.misc.profile
    _batch = yolo.data._batch
    resize_input = yolo.predict.resize_input
    findboxes = yolo.predict.findboxes
    process_box = yolo.predict.process_box


class YOLOv2(framework):
    constructor = yolo.constructor
    parse = yolo.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = yolo.predict.preprocess
    loss = yolov2.train.loss
    is_inp = yolo.misc.is_inp
    postprocess = yolov2.predict.postprocess
    _batch = yolov2.data._batch
    resize_input = yolo.predict.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = yolo.predict.process_box


"""
framework factory
"""

types = {
    '[detection]': YOLO,
    '[region]': YOLOv2
}


def create_framework(meta, FLAGS): #launching a selected framework with meta and FLAGS as args
    net_type = meta['type']
    this = types.get(net_type, framework) #"This" will be assigned the value YOLO/v2 or framework as default
    return this(meta, FLAGS) # Selected framework is called with meta and FLAGS as args


def create_framework(meta, FLAGS):  # launching a selected framework with meta and FLAGS as args
    net_type = meta['type']
    this = types.get(net_type, framework)  # "This" will be assigned the value YOLO/v2 or framework as default
    return this(meta, FLAGS)  # Selected framework is called with meta and FLAGS as args

