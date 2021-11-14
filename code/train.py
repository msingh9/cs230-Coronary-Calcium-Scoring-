import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
from statistics import mean, median, mode, variance
import sys
import xml.etree.ElementTree as et
import plistlib
import matplotlib.patches as patches
import bz2
import gc

# import models
import models.unet as unet

# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
model_name = 'unet'
use_adam = True
learning_rates = [0.0001]
decay_rate = 0
decay_epochs = 10
momentum = 0.9
batch_sizes = [8]
epochs = 5
plot = True
train = True

# Model parameters
params = {}
params['resetHistory'] = False
params['print_summary'] = True
params['dropout'] = 0
params['data_aug_enable'] = False
params['models_dir'] = '../trained_models/' + model_name
params['upsample_ps'] = 40 ; # set non-zero integer to up-sample positive samples
params['limit_pids'] = None
params['alpha'] = 1.0 ; # fraction of dice loss

# data set directory
ddir = "../dataset"

# Read train, dev and test set Ids
fname = ddir + "/gated_train_dev_pids.dump"
with open(fname, 'rb') as fin:
    print(f"Loading train/dev from {fname}")
    train_pids, dev_pids = pickle.load(fin)

fname = ddir + "/gated_test_pids.dump"
with open(fname, 'rb') as fin:
    print(f"Loading test from {fname}")
    test_pids = pickle.load(fin)


print (train_pids)
print (dev_pids)

print (f"Total train samples {len(train_pids)}")
print (f"Total dev samples {len(dev_pids)}")
print (f"Total test samples {len(test_pids)}")

# LossHistory Class
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_seg_f1 = []
        self.val_seg_f1 = []
        self.train_class_acc = []
        self.val_class_acc = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_seg_f1.append(logs.get('seg_f1'))
        self.val_seg_f1.append(logs.get('val_seg_f1'))
        self.train_class_acc.append(logs.get('class_acc'))
        self.val_class_acc.append(logs.get('val_class_acc'))

        gc.collect()
        if epoch%5 == 0:
            # Save model
            print ("Saving the model in ../experiments/current/m_" + str(epoch))
            model.save('../experiments/current/m_' + str(epoch))

# learning rate scheduler
def lr_scheduler(epoch, lr):
    global decay_rate, decay_epochs
    if epoch%decay_epochs == 0 and epoch and decay_rate != 0:
        return lr * decay_rate
    return lr

## Load Model and run training
for batch_size in batch_sizes:
    for lr in learning_rates:
        if train or 'models_dir' not in params:
            params['models_dir'] = f'../experiments/{model_name}/{batch_size}.{lr}'
        history = LossHistory()
        if model_name == 'unet':
            model = unet.Model(history, params)
        else:
            model = None
            exit("Something went wrong, model not defined")

        if not train:
            model.is_train = False

        ## training
        if use_adam:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

        model.compile(optimizer)

        #Set the data
        model.train_pids = train_pids
        model.dev_pids = dev_pids
        model.test_pids = test_pids

        # instantiate model
        if train:
            model.train(batch_size, epochs, lr_scheduler)
            model.save()

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            model.train_plot(fig, ax, show_plot=False)

if plot or train:
    plt.show()

