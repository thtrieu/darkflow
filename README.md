## Intro

This repo aims at building a tensorflow version of Darknet framework, where the famous framework YOLO (real time object detection & classification) is produced. In fact, the goal is to build a framework with `Tensorflow` backend that is compatible with Darknet files, including binary `.weights` and configuration `.cfg` - who looks something like this:


```python
...

[convolutional]
size = 3
stride = 1
pad = 1
activation = leaky

[maxpool]

[connected]
output = 4096
activation = linear

...
```

Imagine design a deep net with such ease! Many thanks to the Darknet author. Currently, `darktf` is built to sufficiently run YOLO. For other net structures, new code will be added. Take a look at `./yolo/` to see how this task should be quite simple.

Regarding bridging Darknet and Tensorflow, there are currently some available repos online such as [_this_](https://github.com/sunshineatnoon/Darknet.keras) and [_this_](https://github.com/gliese581gg/YOLO_tensorflow). Unfortunately, they only provide hard-coded routines that allows translating YOLO's full/small/tiny configurations from Darknet to Tensorflow, and only for testing (forward pass). The awaited training part is still not committed.

This is understandable since building the loss op of YOLO in `Tensorflow` is not a trivial task, it requires careful computational considerations. But hey, I've got time to do that. Namely, we are now able to create new configurations and train them in GPU/CPU mode. Moreover, YOLO would not be completed if it is not running real-time (preferably on mobile devices), `darktf` also allows saving the trained weights to a constant protobuf object that can be used in `Tensorflow` C++ interface.


## How to use it

### Parsing the annotations

Skip this if you are not training or fine-tuning anything (you simply want to forward flow a trained net)

The first thing to do is specifying the classes you want to work with, write them down in the `labels.txt` file. For example, if you want to work with only 3 classes `tvmonitor`, `person`, `pottedplant`; edit `labels.txt` as follows

```
tvmonitor
person
pottedplant
```

Then run `clean.py` to parse xml files in the annotation folder (according to what has been specified in `labels.txt`)

```bash
python clean.py /path/to/annotation/folder
# the default path is ../pascal/VOCdevkit/ANN
```

This will print some stats on the parsed dataset to screen. Parsed bounding boxes and their associated classes is stored in `parsed.yolotf`.

### Design the net

Skip this if you are working with one of the three original configurations since they are already there.

In this step you create a configuration `[config_name].cfg` and put it inside `./configs/`. Take a look at some of the available configs there to know the syntax.

Note that these files, besides being descriptions of the net structures, also store technical specifications that is read by Darknet framework (e.g. learning rate, batch size, epoch number). `darktf` therefore, ignore these Darknet specifications.

### Initialize weights

Skip this if you are working with one of the original configurations since the `.weights` files are already there.

Now as you have already specified the new configuration, next step is to initialize the weights. In this step, it is reasonable to recollect a few first layers from some trained configuration before randomly initialize the rest. `makew.py` does exactly this.

```bash
# Recollect weights from yolo-tiny.weights to yolo-3c.weights
python genw.py --src yolo-tiny --des yolo-3c
```

The script prints out which layers are recollected and which are randomly initialized. The recollected layers are a few first ones that are identical between two configurations. In case there is no such layer, all the new net will be randomly initialized. 

After all this, `yolo-3c.weights` is created. Bear in mind that unlike `yolo-tiny.weights`, `yolo-3c.weights` is not yet trained.

### Flowing the graph

From now on, all operations are performed by file `darktf`. 

```bash
# Have a look at its options
./darktf --h
# Forward all images in ./test using tiny yolo and 100% GPU usage
./darktf --test test --model yolo-tiny --gpu 1.0
# The results are stored in results/
```

Training the new configuration:

```bash
./darktf --train --model yolo-3c --gpu 1.0
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `./backup/`. Only the 20 most recent pairs are kept, you can change this number in the `keep` option, if `keep = 0`, no intermediate result is **omitted**.

To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darktf` will load the most recent save.

```bash
# To resume the most recent checkpoint for training
./darktf --train --model yolo-3c --load -1
# To run testing with checkpoint at step 1500
./darktf --notrain --model yolo-3c --load 1500
# Without the --load option, you will be using the untrained yolo-3c.weights
# Fine tuning tiny yolo from the original one
./darktf --train --model yolo-tiny
```

### Migrating the graph to C++ and Objective-C++

Now this is the tricky part since there is no official support for loading variables in C++ API. Some suggest adding assigning ops from variable to constants into the graph and save it down as a `.pb` (protobuf) file [_like this_](https://alexjoz.gitbooks.io/code-life/content/chapter7.html). However this will double the necessary size of this file, which is very undesirable in, say, building mobile applications. 

There is an official way to do the same thing using [this script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) provided in `Tensorflow`, however doing so would require running on a separate script. `darktf` allows freezing the graph on the fly, during training or testing, without double the necessary size.

```bash
## Saving the lastest checkpoint to protobuf file
./darktf --model yolo-3c --load -1 --savepb
```

For further usage of this protobuf file, please refer to the official documentation of `Tensorflow` on C++ API [_here_](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html). To run it on, say, iOS application, simply add the file to Bundle Resources and update the path to this file inside source code.

That's all!