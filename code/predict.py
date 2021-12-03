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
from my_lib import process_xml, compute_agatston_for_slice
import argparse
from dataGenerator import dataGenerator

# import models
import models.unet as unet
import models.unet1 as unet1
import models.uneta as uneta

loss_choices = ("bce", "dice", "focal", "dice_n_bce")
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, action='append', help="List of batch sizes")
parser.add_argument("-ddir", default="../dataset", type=str, help="Data set directory. Don't change sub-directories of the dataset")
parser.add_argument("-mdir", default="../trained_models/unet", type=str, help="Model's directory")
parser.add_argument("-mname", default="unet", type=str, help="Model's name")
parser.add_argument("-pid", default=0, type=int, help="pid to plot")
parser.add_argument("-loss", type=str, choices=loss_choices, default='dice', help=f"Pick loss from {loss_choices}")
parser.add_argument("-dice_loss_fraction", default=1.0, type=float, help="Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider. Set it to 1.0 to ignore class loss")
parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate the model")
parser.add_argument("-set", type=str, choices=("train", "dev", "test"), default='dev', help="Specify set (train|dev|test) to evaluate or predict on")
parser.add_argument("--print_stats", action="store_true", default=False, help="Predict on all patients and print number of predicted calcified pixels")
parser.add_argument("--only_use_pos_images", action="store_true", default=False, help="Evaluate with positive images only")
parser.add_argument("-pmask_threshold", default=0, type=int, help="A non-zero number will filter lesion less than this area")
parser.add_argument("--print_agatston_score", action="store_true", default=False, help="Print Agaston score for actual and predicted")
args = parser.parse_args()
# User options
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_name = args.mname
batch_size = 8

# Model parameters
params = {}
params['reset_history'] = False ; # Keep this false
params['models_dir'] = args.mdir
params['loss'] = args.loss
params['print_summary'] = False
params['alpha'] = args.dice_loss_fraction ; # fraction of dice loss


# data set directory
ddir = args.ddir

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

if (ddir == "../mini_dataset"):
    ddir = ddir + "/Gated_release_final"
else:
    ddir = ddir + "/cocacoronarycalciumandchestcts-2/Gated_release_final"

# Load Model
if model_name == 'unet':
    model = unet.Model(None, params)
elif model_name == 'unet1':
    model = unet1.Model(None, params)
elif model_name == 'uneta':
    model = uneta.Model(None, params)
else:
    model = None
    exit("Something went wrong, model not defined")

model.is_train = False
# We may not need this?
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer)
print (train_pids)
print (dev_pids)
print (test_pids)

if args.evaluate:
    if args.set == "train":
        print ("Evaluating on train set")
        Y_hat = model.my_evaluate(train_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    elif args.set == "dev":
        print("Evaluating on dev set")
        Y_hat = model.my_evaluate(dev_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    elif args.set == "test":
        print("Evaluating on test set")
        Y_hat = model.my_evaluate(test_pids, batch_size, only_use_pos_images=args.only_use_pos_images)
    else:
        exit(f"ERROR: Unknown {args.eval_set}")
    exit()


if args.print_stats:
    stats = []
    for pid in dev_pids:
        Y_hat = model.my_predict([pid], batch_size)
        stats.append((pid, np.sum(Y_hat > 0.5)))

    for _ in stats:
        print (_)
    exit()


ag_scores = []
pids = [args.pid]
if args.print_agatston_score:
    if args.set == "train":
        pids = train_pids
    elif args.set == "dev":
        pids = dev_pids
    elif args.set == "test":
        pids = test_pids
    else:
        exit(f"ERROR: Unknown {args.eval_set}")

for pid in pids:
    # Plot original and prediction for given pid
    Y_hat = model.my_predict([pid], batch_size)
    plot_3d = True

    images = []
    for subdir, dirs, files in os.walk(ddir + "/patient/" + str(pid) + '/'):
        for filename in sorted(files, reverse=True):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".dcm"):
                ds = dcmread(filepath)
                images.append(ds.pixel_array)

    images = np.array(images)

    # read original mdata
    fname = ddir + "/calcium_xml/" + str(pid) + (".xml")
    if os.path.exists(fname):
        mdata = process_xml(fname)
    else:
        mdata = None

    # Get True Y
    my_dg = dataGenerator([pid], batch_size, shuffle=False)
    m, height, width, _ = Y_hat.shape
    Y_true = np.zeros(Y_hat.shape)
    X_all = np.zeros((m, height, width))
    for i in range(len(my_dg)):
        if ((i + 1) * batch_size) < Y_true.shape[0]:
            Y_true[i * batch_size : (i+1) * batch_size] = my_dg[i][1]
            X_all[i * batch_size : (i+1) * batch_size] = my_dg[i][0].reshape(-1, height, width)
        else:
            Y_true[i * batch_size : ] = my_dg[i][1]
            X_all[i * batch_size:] = my_dg[i][0].reshape(-1, height, width)

    # Compute Agatson score from true mask
    ag_score_true = 0
    for X, Y in zip(X_all, Y_true):
        ag_score_true += compute_agatston_for_slice(X, Y)

    # Create mdata from prediction
    #  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
    pmdata = {}
    ymdata = {}
    ## FIXME, extract the predicted cid
    ag_score_hat = 0
    for id in range(Y_hat.shape[0]):
        pred = Y_hat[id][:, :, 0] > 0.5
        ag_score_hat += compute_agatston_for_slice(X_all[id], pred.reshape(height, width, 1))
        Y, X = np.where(pred)
        print (np.sum(pred))
        if len(Y) > 0:
            pmdata[id] = []
            ttt = {'cid': 0, 'pixels': []}
            for y, x in zip(Y, X):
                ttt['pixels'].append((x,y))
            pmdata[id].append(ttt)

    ag_scores.append((pid, ag_score_true, ag_score_hat))
    if args.print_agatston_score:
        continue

    #print (Y_hat.shape)
    #print (Y_true.shape)
    #print (model.model.get_weights())

    for id in range(Y_hat.shape[0]):
        Y, X = np.where(Y_true[id][:, :, 0] > 0)
        if len(Y) > 0:
            ymdata[id] = []
            ttt = {'cid': 0, 'pixels': []}
            for y, x in zip(Y, X):
                ttt['pixels'].append((x,y))
            ymdata[id].append(ttt)

    #pmdata = ymdata

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


if args.print_agatston_score:
    for ag_score in ag_scores:
        print (",".join([str(_) for _ in ag_score]))
exit()
