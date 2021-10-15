
import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import pickle
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
from statistics import mean, median, mode, variance
import sys
import xml.etree.ElementTree as et
import plistlib


# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

# for debug messages
def myprint(msg, level):
    if level <= debug:
        print (msg)

# data path directory
ddir = "D:/cs230/accs/dataset/cocacoronarycalciumandchestcts-2"
#ddir = "D:/cs230/accs/mini_dataset"
sdirs = {"G" : "Gated_release_final",
           "N" : "deidentified_nongated"}

# directory structure
# G/calcium_xml/<id>
# G/patient/<id>/*/*.dcm
# NG/<id>/<id>/*.dcm
# NG/scores (csv)

# read all images of a given patient
# this is to plot CT for a given patient
# for 3D plots:
    # p -> to go to previous slice
    # n -> to go to next slice

doPlot = False
plot3D = False
images = []
pid = 1 # specify the patient id
sdir_id = "N" ;# G for gated, N for non-gated

if doPlot:
    if sdir_id == "G":
        pdir = '%s/%s/patient/%s/' %(ddir, sdirs[sdir_id], pid)
    else:
        pdir = '%s/%s/%s/' % (ddir, sdirs[sdir_id], pid)
    for subdir, dirs, files in os.walk(pdir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".dcm"):
                ds = dcmread(filepath)
                images.append(ds.pixel_array)

    images = np.array(images)

    # plot
    if plot3D:
        def previous_slice(ax):
            """Go to the previous slice."""
            volume = ax.volume
            n = volume.shape[0]
            ax.index = (ax.index - 1) % n  # wrap around using %
            ax.images[0].set_array(volume[ax.index])

        def next_slice(ax):
            """Go to the next slice."""
            volume = ax.volume
            n = volume.shape[0]
            ax.index = (ax.index + 1) % n
            ax.images[0].set_array(volume[ax.index])
            pass

        def process_key(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == 'p':
                previous_slice(ax)
            elif event.key == 'n':
                next_slice(ax)
            fig.canvas.draw()


        def multi_slice_viewer(volume):
            fig, ax = plt.subplots()
            ax.volume = volume
            ax.index = volume.shape[0] // 2
            ax.imshow(volume[ax.index], cmap='gray')
            fig.canvas.mpl_connect('key_press_event', process_key)

        multi_slice_viewer(images)
        plt.show()
    else:
        fig = plt.figure(figsize=(512,512))
        s = int(images.shape[0]**0.5 + 0.99)
        grid = ImageGrid(fig, 111, nrows_ncols = (s,s))
        for i in range(images.shape[0]):
            ax = grid[i]
            ax.imshow(images[i,:,:], cmap="gray")

        plt.show()

    exit()

data = {}
# {G: {<pid>: {images: [], mdata: {}},
# {N: {<pid>: {images: [], mdata: {}}}

# to track progress
total_work = 0
progress_count = 0

def tick():
    global progress_count, total_work
    progress_count += 1
    if (progress_count%64 == 0):
        p = int(progress_count*100/total_work)
        sys.stdout.write(f"\r{p}%")
        sys.stdout.flush()

# Process gated CTs
k = "G"
data[k] = {}
sdir = f"{ddir}/{sdirs[k]}/patient"
myprint(f"Processing {sdir} folder", 1)
for subdir, dirs, files in os.walk(sdir):
    total_work = len(files)
    progress_count = 0
    for filename in files:
        tick()
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        myprint(f"Processing {filepath}", 4)
        if filepath.endswith(".dcm"):
            pid = filepath.split("/")[-3]
            if (not pid in data[k]):
                data[k][pid] = {}
                data[k][pid]['images'] = []
            data[k][pid]['images'].append(dcmread(filepath))

# process calcium_xml
def process_xml(f):
    # input XML file
    # output - directory containing various meta data
    with open(f, 'rb') as fin:
        pl = plistlib.load(fin)

sdir = f"{ddir}/{sdirs[k]}/calcium_xml"
myprint(f"Processing {sdir} folder", 1)
for subdir, dirs, files in os.walk(sdir):
    for filename in files:
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        myprint(f"Processing {filepath}", 4)
        if filepath.endswith(".xml"):
            pid = filepath.split("/")[-1].split(".")[0]
            if (pid in data[k]):
                data[k][pid]['mdata'] = process_xml(filepath)
            else:
                print (f"WARNING: {pid}.xml found but no matching images")

# process non-gated CTs
k = "N"
data[k] = {}
sdir = f"{ddir}/{sdirs[k]}"
myprint(f"Processing {sdir}/{sdir} folder", 1)
for subdir, dirs, files in os.walk(sdir):
    total_work = len(files)
    progress_count = 0
    for filename in files:
        tick()
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        myprint(f"Processing {filepath}", 4)
        if filepath.endswith(".dcm"):
            pid = filepath.split("/")[-3]
            if (not pid in data[k]):
                data[k][pid] = {}
                data[k][pid]['images'] = []
            data[k][pid]['images'].append(dcmread(filepath))

# process non-gated score file
with open(sdir + "/scores.csv") as fin:
    csvreader = csv.reader(fin)
    is_header = True
    for row in csvreader:
        if is_header:
            is_header = False
            continue

        pid, lca, lad, lcx, rca, total = row
        pid = pid.rstrip("A")
        lca = float(lca)
        lad = float(lad)
        lcx = float(lcx)
        rca = float(rca)
        total = float(total)
        # FIXME: hmm, what is total?
        #sum = lca + lad + lcx + rca
        #assert(total > sum - 1 and total < sum + 1), f"TOTAL doesn't match ({total} != {lca} + {lad} + {lcx} + {rca})"
        if (pid in data[k]):
            data[k][pid]['mdata'] = [lca, lad, lcx, rca, total]
        else:
            print(f"WARNING: {pid}.xml found but no matching images")

# print stats about data
info = {}
info["G"] = {}
info["N"] = {}
ng_scores = []
for k in ("N", "G"):
    info[k]['num_of_patients'] = 0
    info[k]['num_of_slices'] = 0
    info[k]['num_of_pos_slices'] = 0
    info[k]['num_of_pos_patients'] = 0
    for pid in data[k].keys():
        info[k]['num_of_patients'] += 1
        info[k]['num_of_slices'] += len(data[k][pid]['images'])
        if k == "N":
            if (pid in data[k] and 'mdata' in data[k][pid] and data[k][pid]['mdata'][-1]):
                info[k]['num_of_pos_patients'] += 1
                info[k]['num_of_pos_slices'] += 1
                ng_scores.append(data[k][pid]['mdata'][-1])
        else:
            if (pid in data[k] and 'mdata' in data[k][pid]):
                info[k]['num_of_pos_patients'] += 1
                info[k]['num_of_pos_slices'] += 1 #FIXME: we need to process xml data


# print stats
for k in ("N", "G"):
    print (f"{sdirs[k]} statistics")
    print (info[k])
print ("Nogated scores distribution")
print ("Min = %.2f, Max = %.2f, Mean = %.2f" %(min(ng_scores), max(ng_scores), mean(ng_scores)))
print ("Median = %.2f, Variance = %.2f" %(median(ng_scores), variance(ng_scores)))