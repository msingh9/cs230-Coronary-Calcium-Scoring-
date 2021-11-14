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
from process_xml import process_xml

# import models
import models.unet as unet

# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_name = 'unet'
batch_size = 8

# Model parameters
params = {}
params['reset_history'] = False ; # Keep this false
params['models_dir'] = '../trained_models/' + model_name

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

print (f"Total train samples {len(train_pids)}")
print (f"Total dev samples {len(dev_pids)}")
print (f"Total test samples {len(test_pids)}")

# Load Model
if model_name == 'unet':
    model = unet.Model(None, params)
else:
    model = None
    exit("Something went wrong, model not defined")

model.is_train = False
# We may not need this?
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer)

#Set the data
model.dev_pids = dev_pids
model.test_pids = test_pids

# Plot original and prediction
print (train_pids)
print (dev_pids)
plot_pid = 113
Y_hat = model.my_predict([plot_pid], batch_size)
print (Y_hat.shape)

plot_3d = True

images = []
for subdir, dirs, files in os.walk(ddir + "/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/" + str(plot_pid) + '/'):
    for filename in sorted(files, reverse=True):
        filepath = subdir + os.sep + filename
        if filepath.endswith(".dcm"):
            ds = dcmread(filepath)
            images.append(ds.pixel_array)

images = np.array(images)

# read original mdata
fname = ddir + "/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml/" + str(plot_pid) + (".xml")
if os.path.exists(fname):
    mdata = process_xml(fname)
else:
    mdata = None

# Create mdata from prediction
#  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
pmdata = {}
## FIXME, extract the predicted cid
for id in range(Y_hat.shape[0]):
    X, Y = np.where(Y_hat[id][:, :, 0] > 0.5)
    if len(Y) > 0:
        pmdata[id] = []
        ttt = {'cid': 0, 'pixels': []}
        for y, x in zip(Y, X):
            ttt['pixels'].append((x,y))
        pmdata[id].append(ttt)

# plot
pixel_colors = {0: 'red',
                1: 'blue',
                2: 'green',
                3: 'yellow'}

def add_patches(ax, mdata, is_predict=False):
    ax.patches = []
    if mdata and plot_cid in mdata:
        for roi in mdata[plot_cid]:
            if (is_predict):
                for p in roi['pixels']:
                    ax.add_patch(patches.Circle(p, radius=1, color=pixel_colors[roi['cid']]))
            else:
                ax.add_patch(patches.Polygon(roi['pixels'], closed=True, color=pixel_colors[roi['cid']]))

if plot_3d:
    def previous_slice(ax):
        """Go to the previous slice."""
        global plot_cid
        volume = ax[0].volume
        n = volume.shape[0]
        plot_cid = (plot_cid - 1) % n  # wrap around using %
        for i in range(2):
            ax[i].images[0].set_array(volume[plot_cid])
            ax[i].set_title(f"Image {plot_cid}")
        add_patches(ax[0], mdata)
        add_patches(ax[1], pmdata, is_predict=True)


    def next_slice(ax):
        """Go to the next slice."""
        global plot_cid
        volume = ax[0].volume
        n = volume.shape[0]
        plot_cid = (plot_cid + 1) % n
        for i in range(2):
            ax[i].images[0].set_array(volume[plot_cid])
            ax[i].set_title(f"Image {plot_cid}")
        add_patches(ax[0], mdata)
        add_patches(ax[1], pmdata, is_predict=True)


    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes
        if event.key == 'p':
            previous_slice(ax)
        elif event.key == 'n':
            next_slice(ax)
        fig.canvas.draw()


    def multi_slice_viewer(volume):
        global plot_cid
        fig, ax = plt.subplots(1, 2)
        ax[0].volume = volume
        plot_cid = volume.shape[0] // 2
        img = volume[plot_cid]
        for i in range(2):
            ax[i].imshow(img, cmap="gray")
            ax[i].set_title(f"Image {plot_cid}")
        add_patches(ax[0], mdata)
        add_patches(ax[1], pmdata, is_predict=True)
        fig.canvas.mpl_connect('key_press_event', process_key)


    multi_slice_viewer(images)
    plt.show()
else:
    fig = plt.figure(figsize=(512, 512))
    s = int(images.shape[0] ** 0.5 + 0.99)
    grid = ImageGrid(fig, 111, nrows_ncols=(s, s))
    for i in range(images.shape[0]):
        ax = grid[i]
        ax.imshow(images[i, :, :], cmap="gray")

    plt.show()

exit()
