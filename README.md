## Update

TODO: yolov2 with new layers like `batch_norm` or `local`.

## Intro

This repo aims at building a tensorflow version of [darknet framework](https://github.com/pjreddie/darknet), where the famous framework [YOLO](http://pjreddie.com/darknet/yolo/) (real time object detection & classification) is created. In fact, the goal is to build a framework with `Tensorflow` backend that is compatible with Darknet files, including binary `.weights` and configuration `.cfg` - who looks something like this:


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

Imagine design a deep net with such ease! Many thanks to the Darknet author. The ease, however, is not complete though - imagine changing the number of classes and you have to do the engineer work: edit some code, rebuild `darknet`. Fortunately you won't have to do so with `darktf`. Currently, `darktf` is built to sufficiently run YOLO. For other net structures, new code will be added in a plug-and-play manner. Take a look at `./tfnet/yolo/` and see how this task should be quite simple.

Regarding bridging Darknet and Tensorflow for YOLO, there are currently some available repos online such as [_this_](https://github.com/sunshineatnoon/Darknet.keras) and [_this_](https://github.com/gliese581gg/YOLO_tensorflow). Unfortunately, they only provide hard-coded routines that allows translating YOLO's full/small/tiny configurations from Darknet to Tensorflow, and only for testing (forward pass). The awaited training part is still not committed.

This is understandable since building the loss op of YOLO in `Tensorflow` is not a trivial task, it requires careful computational considerations. But hey, I've got time to do that. Namely, we are now able to create new configurations and train them in GPU/CPU mode. Moreover, YOLO would not be completed if it is not running real-time (preferably on mobile devices), `darktf` also allows saving the trained weights to a constant protobuf object that can be used in `Tensorflow` C++ interface.


## How to use it

### Parsing the annotations

Skip this if you are not training or fine-tuning anything (you simply want to forward flow a trained net)

The only thing to do is specifying the classes you want to work with, write them down in the `labels.txt` file. For example, if you want to work with only 3 classes `tvmonitor`, `person`, `pottedplant`; edit `labels.txt` as follows

```
tvmonitor
person
pottedplant
```

And that's it. `darktf` will parse the annotation if necessary.

### Design the net

Skip this if you are working with one of the three original configurations since they are already there.

In this step you create a configuration `[config_name].cfg` and put it inside `./configs/`. Take a look at some of the available configs there to know the syntax.

Note that these files, besides being descriptions of the net structures, also store technical specifications that is read by Darknet framework (e.g. learning rate, batch size, epoch number). `darktf` therefore, ignore these Darknet specifications.

### Flowing the graph

From now on, all operations are performed by file `flow`. 

```bash
# Have a look at its options
./flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. With no --load option, yolo-tiny.weights are loaded
./flow --model ./configs/yolo-tiny.cfg --load ./bin/yolo-tiny.weights
# 2. With yolo-3c however, since there are no yolo-3c.weights,
# its parameters will be randomly initialized
./flow --model ./configs/yolo-3c.cfg
# 3. It is useful to reuse the first identical layers of tiny for 3c
./flow --model ./configs/yolo-3c.cfg --load ./bin/yolo-tiny.weights
# this will print out which layers are reused, which are initialized
```

More on `--load` later. All of the above `flow` commands essentially perform forward pass of the net. In fact, they flow all input images from default folder `./test` through the net and draw predictions into `./results/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, test folder, etc. Below is one example where the forward pass is told to utilize 100% GPU capacity:

```bash
# Forward all images in ./test using tiny yolo and 100% GPU usage
./flow --test ./test --model ./configs/yolo-tiny.cfg --load ./bin/yolo-tiny.weights --gpu 1.0
# The results are stored in results/
```

Training is simple as you only have to add option `--train` like below:

```bash
# Initialize yolo-3c from yolo-tiny, then train the net on 100% GPU:
./flow --model ./configs/yolo-3c.cfg --load ./bin/yolo-tiny.weights --train --gpu 1.0
# Completely initialize yolo-3c and train it with ADAM optimizer
./flow --model ./configs/yolo-3c.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `./backup/`. Only the 20 most recent pairs are kept, you can change this number in the `keep` option, if `keep = 0`, no intermediate result is **omitted**.

To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darktf` will load the most recent save. Here are a few examples:

```bash
# To resume the most recent checkpoint for training
./flow --train --model ./configs/yolo-3c.cfg --load -1
# To run testing with checkpoint at step 1500
./flow --model ./configs/yolo-3c.cfg --load 1500
# Fine tuning tiny yolo from the original one
./flow --train --model ./configs/yolo-tiny.cfg --load ./bin/yolo-tiny.weights
```

You can even initialize a new net by `ckpt` file with `--load`:
```bash
./flow --train --model ./configs/yolo-2c.cfg --load ./backup/yolo-3c-1500
# recollected and initialized layers will be printed to console
```

### Migrating the graph to C++ and Objective-C++

Now this is the tricky part since there is no official support for loading variables in C++ API. Some suggest adding assigning ops from variable to constants into the graph and save it down as a `.pb` (protobuf) file [_like this_](https://alexjoz.gitbooks.io/code-life/content/chapter7.html). However this will double the necessary size of this file (or even triple if there is training ops), which is very undesirable in, say, building mobile applications. 

There is an official way to do the same thing using [this script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) provided in `Tensorflow`. I did not have the time to check its implementation and performance, however doing so would certainly require running on a separate script. `darktf` allows freezing the graph on the fly, during training or testing, without doubling/tripling the necessary size.

```bash
## Saving the lastest checkpoint to protobuf file
./flow --model ./configs/yolo-3c.cfg --load -1 --savepb
```

For further usage of this protobuf file, please refer to the official documentation of `Tensorflow` on C++ API [_here_](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html). To run it on, say, iOS application, simply add the file to Bundle Resources and update the path to this file inside source code.

That's all!