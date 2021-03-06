import argparse
import datetime
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
import tensorflow as tf
import random
import os
import pytz
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
import models.unet1 as unet1
import models.uneta as uneta

loss_choices = ("bce", "dice", "focal", "dice_n_bce")

parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, action='append', help="List of batch sizes")
parser.add_argument("-epochs", default=5, type=int)
parser.add_argument("-max_train_patients", default=None, type=int, help="To limit number of training examples")
parser.add_argument("-dice_loss_fraction", default=1.0, type=float, help="Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider. Set it to 1.0 to ignore class loss")
parser.add_argument("-upsample_ps", default=None, type=int, help="Non zero value to enable up-sampling positive samples during training")
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-patient_splits_dir", type=str, help="Directory in which patient splits are located.", default=None)
parser.add_argument("-mdir", default="../trained_models/unet", type=str, help="Model's directory")
parser.add_argument("-mname", default="unet", type=str, help="Model's name")
parser.add_argument("--plot", action="store_true", default=True, help="Plot the metric/loss")
parser.add_argument("--train", action="store_true", default=False, help="Train the model")
parser.add_argument("--hsen", action="store_true", default=False, help="Generate random hyper parameters")
parser.add_argument("-lr", help="List of learning rates", action='append', type=float)
parser.add_argument("-steps_per_epoch", default=None, type=int, help="Number of steps per epoch. Set this to increase the frequency at which Tensorboard reports eval metrics. If None, it will report eval once per epoch.")
parser.add_argument("-model_save_freq_steps", default=None, type=int,
                    help="Save the model at the end of this many batches. If low,"
                    "can slow down training. If none, save after each epoch.")
parser.add_argument("-loss", type=str, choices=loss_choices, default='dice', help=f"Pick loss from {loss_choices}")
parser.add_argument("--reset", default=False, action="store_true", help="To reset model")
parser.add_argument("--only_use_pos_images", action="store_true", default=False, help="Train with positive images only")
parser.add_argument("--use_dev_pos_images", action="store_true", default=False, help="Evaluate only on positive samples on dev set")
parser.add_argument("--den", action="store_true", default=False, help="Enable data augmentation")
parser.add_argument("-num_neg_images_per_batch", default=0, type=int, help="Number of positive images to be replaced with neg images per batch. Use with --only_use_pos_images")
args = parser.parse_args()

TIME_FORMAT = "%Y-%m-%d-%H-%M"

## My overwrite
if 0:
    args.batch_size = [8]
    args.loss = 'focal'
    args.train = True
    args.lr = [0.0001]
    args.model_save_freq_steps = 1
    args.reset = False
    args.epochs = 10
    args.upsample_ps = 8
    args.mname = 'uneta'

def get_time():
  return datetime.datetime.now(pytz.timezone('US/Pacific'))
start_time = get_time()
print(f'Launched at {start_time}')

# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
model_name = args.mname
use_adam = True
if args.lr:
    learning_rates = args.lr
else:
    learning_rates = [0.0001]
decay_rate = 1.0
decay_epochs = 10
momentum = 0.9
batch_sizes = args.batch_size
epochs = args.epochs
plot = args.plot
train = args.train

# Model parameters
params = {}
params['resetHistory'] = args.reset
params['print_summary'] = False
params['dropout'] = 0
params['data_aug_enable'] = args.den
params['models_dir'] = args.mdir
params['upsample_ps'] = args.upsample_ps ; # set non-zero integer to up-sample positive samples
params['limit_pids'] = args.max_train_patients
params['alpha'] = args.dice_loss_fraction ; # fraction of dice loss
params['ddir'] = args.ddir
params['steps_per_epoch'] = args.steps_per_epoch
params['model_save_freq_steps'] = args.model_save_freq_steps
params['loss'] = args.loss
params['only_use_pos_images'] = args.only_use_pos_images
params['use_dev_pos_images'] = args.use_dev_pos_images
params['num_neg_images_per_batch'] = args.num_neg_images_per_batch

# Hyper parameter search
if args.hsen:
    #learning_rates = [10**random.uniform(-2,-5)]
    #print (learning_rates)
    params['alpha'] = random.uniform(0.8, 1.0)
    print (f"Using alpha as {params['alpha']}")

if args.patient_splits_dir is None:
  patient_splits_dir = args.ddir
else:
  patient_splits_dir = args.patient_splits_dir

# Read train, dev and test set Ids
fname = os.path.join(patient_splits_dir, "gated_train_dev_pids.dump")
with open(fname, 'rb') as fin:
    print(f"Loading train/dev from {fname}")
    train_pids, dev_pids = pickle.load(fin)

fname = os.path.join(patient_splits_dir, "gated_test_pids.dump")
with open(fname, 'rb') as fin:
    print(f"Loading test from {fname}")
    test_pids = pickle.load(fin)

#print (train_pids)
#print (dev_pids)

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

    def on_epoch_end(self, epoch, logs):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_seg_f1.append(logs.get('seg_f1'))
        self.val_seg_f1.append(logs.get('val_seg_f1'))
        self.train_class_acc.append(logs.get('class_acc'))
        self.val_class_acc.append(logs.get('val_class_acc'))

        gc.collect()
        if args.model_save_freq_steps and epoch % args.model_save_freq_steps == 0:
            # Save model
            print ("Saving the model in ../experiments/current/m_" + str(epoch))
            model.save('../experiments/current/m_' + str(epoch))
            #print ("Full evaluation on dev set")
            #model.my_evaluate(dev_pids, batch_size, only_use_pos_images=False)

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
        elif model_name == 'unet1':
            model = unet1.Model(history, params)
        elif model_name == 'uneta':
            model = uneta.Model(history, params)
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
            #y_hat = model.my_evaluate(train_pids, batch_size, only_use_pos_images=args.only_use_pos_images)

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            model.train_plot(fig, ax, show_plot=False)

end_time = get_time()
elapsed_time = (end_time - start_time).total_seconds() / 3600
print(f'Completed at {end_time}. Elapsed hours: {elapsed_time}')
if plot or train:
    plt.show()

