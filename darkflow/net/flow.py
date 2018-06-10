import os
import time
import numpy as np
import tensorflow as tf
import pickle
from subprocess import call
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)

    print("Upload model ckpt")
    bucket_path = os.path.join(self.FLAGS.bucket, "models_4")
    local_path = os.path.join(self.FLAGS.backup)
    call(["gsutil", "-m", "rsync", "-r", local_path, bucket_path])

    print("Upload tensorboard for train")

    bucket_path = os.path.join(self.FLAGS.bucket, "tb_train_4")
    local_path = os.path.join(self.FLAGS.summary, "train")
    call(["gsutil", "-m", "rsync", "-r", local_path, bucket_path])

    print("Upload tensorboard for valid")

    bucket_path = os.path.join(self.FLAGS.bucket, "tb_val_4")
    local_path = os.path.join(self.FLAGS.val_summary, "val")
    call(["gsutil", "-m", "rsync", "-r", local_path, bucket_path])

    print("Finished saving checkpoint")



def train(self):

    arg_steps = self.FLAGS.steps[1:-1] # remove '[' and ']'
    arg_steps = arg_steps.split(',')
    arg_steps = np.array(arg_steps).astype(np.int32)
    arg_scales = self.FLAGS.scales[1:-1]  # remove '[' and ']'
    arg_scales = arg_scales.split(',')
    arg_scales = np.array(arg_scales).astype(np.float32)
    lr = self.FLAGS.lr

    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()
    loss_mva_valid = None

    batches = self.framework.shuffle()
    val_batches = self.framework.shuffle(training = False)
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        feed_dict[self.learning_rate] = lr
        idx = np.where(arg_steps[:] == i + 1)[0]
        if len(idx):
            new_lr = lr * arg_scales[idx][0]
            lr = new_lr
            feed_dict[self.learning_rate] = lr
            print("\nSTEP {} - UPDATE LEARNING RATE TO {:.6}".format(i+1, new_lr))
        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

        #validation time
        if i % self.FLAGS.val_steps == 0:
            (x_batch, datum) = next(val_batches)
            feed_dict = {
                loss_ph[key]: datum[key] 
                    for key in loss_ph }
            feed_dict[self.inp] = x_batch
            feed_dict.update(self.feed)
            feed_dict[self.learning_rate] = lr

            fetches = [loss_op, self.summary_op] 
            fetched = self.sess.run(fetches, feed_dict)
            loss = fetched[0]

            if loss_mva_valid is None: loss_mva_valid = loss
            loss_mva_valid = .9 * loss_mva_valid + .1 * loss

            self.val_writer.add_summary(fetched[1], step_now)

            form = 'VALIDATION step {} - loss {} - moving ave loss {}'
            self.say(form.format(step_now, loss, loss_mva_valid))




    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool = ThreadPool()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
