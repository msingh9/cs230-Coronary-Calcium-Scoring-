
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
import matplotlib.patches as patches
import bz2
from process_xml import process_xml

# User options
doPlot = False
plot3D = True
pid = random.choice([i for i in range(100)]) # specify the patient id
pid = 100
print (pid)
sdir_id = "G" ;# G for gated, N for non-gated
generate_gated_train_dev_test_set = True
train_set_size = 0.8
dev_set_size = 0.1
test_set_size = 0.1

# data path directory
ddir = "../dataset/cocacoronarycalciumandchestcts-2"
sdirs = {"G" : "Gated_release_final",
           "N" : "deidentified_nongated"}

# output data directory where pickle objects will be dumped.
odir = "../dataset"

# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

# myprint function
def myprint(x, level):
    if (level < debug):
        print (x)

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

images = []

# pixel colors
pixel_colors = {0: 'red',
                1: 'blue',
                2: 'green',
                3: 'yellow'}

# function to add patches from mdata on given matplot Axes
plot_cid = 0 ;# global variable to remember image index that is currently plotted
def add_patches(ax, mdata):
    ax[1].patches = []
    if plot_cid in mdata:
        for roi in mdata[plot_cid]:
            ax[1].add_patch(patches.Polygon(roi['pixels'], closed=True, color=pixel_colors[roi['cid']]))
if doPlot:
    if sdir_id == "G":
        pdir = '%s/%s/patient/%s/' %(ddir, sdirs[sdir_id], pid)
    else:
        pdir = '%s/%s/%s/' % (ddir, sdirs[sdir_id], pid)
    for subdir, dirs, files in os.walk(pdir):
        for filename in sorted(files, reverse=True):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".dcm"):
                ds = dcmread(filepath)
                images.append(ds.pixel_array)

    images = np.array(images)
    if sdir_id == "G":
        fname = "%s/%s/calcium_xml/%s.xml" %(ddir, sdirs[sdir_id], pid)
        myprint(f"Processing {fname}", 2)
        mdata = process_xml(fname)
        # for k,v in mdata.items():
        #     print (k)
        #     for temp in v:
        #         for k1, v1 in temp.items():
        #             print (k1, v1)
    else:
        mdata = None

    # plot
    if plot3D:
        def previous_slice(ax):
            """Go to the previous slice."""
            global plot_cid
            volume = ax[0].volume
            n = volume.shape[0]
            plot_cid = (plot_cid - 1) % n  # wrap around using %
            for i in range(2):
                ax[i].images[0].set_array(volume[plot_cid])
                ax[i].set_title(f"Image {plot_cid}")
            add_patches(ax, mdata)

        def next_slice(ax):
            """Go to the next slice."""
            global plot_cid
            volume = ax[0].volume
            n = volume.shape[0]
            plot_cid = (plot_cid + 1) % n
            for i in range(2):
                ax[i].images[0].set_array(volume[plot_cid])
                ax[i].set_title(f"Image {plot_cid}")
            add_patches(ax, mdata)

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
            fig, ax = plt.subplots(1,2)
            ax[0].volume = volume
            plot_cid = volume.shape[0] // 2
            img = volume[plot_cid]
            for i in range(2):
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(f"Image {plot_cid}")
            add_patches(ax, mdata)
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

# estimate total work
total_work = 0
for subdir, dirs, files in os.walk(sdir):
    total_work += len(files)

progress_count = 0
for subdir, dirs, files in os.walk(sdir):
    images_indices = []
    for filename in sorted(files, reverse=True):
        tick()
        filepath = str.replace(subdir, "\\", "/") + "/" + filename
        image_index = filename.split(".")[0].split("-")[-1]
        if image_index in images_indices:
            print (f" duplicate images in {filepath}")
            break
        images_indices.append(image_index)
        myprint(f"Processing {filepath}", 4)
        if filepath.endswith(".dcm"):
            pid = filepath.split("/")[-3]
            if (not pid in data[k]):
                data[k][pid] = {}
                data[k][pid]['images'] = []
            data[k][pid]['images'].append(dcmread(filepath))

sdir = f"{ddir}/{sdirs[k]}/calcium_xml"
myprint(f"Processing {sdir} folder", 1)

# estimate total work
total_work = 0
for subdir, dirs, files in os.walk(sdir):
    total_work += len(files)

progress_count = 0
for subdir, dirs, files in os.walk(sdir):
    for filename in files:
        tick()
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

# estimate total work
total_work = 0
for subdir, dirs, files in os.walk(sdir):
    total_work += len(files)

progress_count = 0
for subdir, dirs, files in os.walk(sdir):
     for filename in sorted(files, reverse=True):
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
            if (pid in data[k] and 'mdata' in data[k][pid] and data[k][pid]['mdata'][-1] > 0):
                info[k]['num_of_pos_patients'] += 1
                info[k]['num_of_pos_slices'] += 1
                ng_scores.append(data[k][pid]['mdata'][-1])
        else:
            if (pid in data[k] and 'mdata' in data[k][pid]):
                if len(data[k][pid].keys()) > 0:
                    info[k]['num_of_pos_patients'] += 1
                for key in data[k][pid].keys():
                    info[k]['num_of_pos_slices'] += 1


# print stats
for k in ("N", "G"):
    print (f"{sdirs[k]} statistics")
    print (info[k])
print ("Nogated scores distribution")
print ("Min = %.2f, Max = %.2f, Mean = %.2f" %(min(ng_scores), max(ng_scores), mean(ng_scores)))
print ("Median = %.2f, Variance = %.2f" %(median(ng_scores), variance(ng_scores)))

# train/test/dev split
all_pids = []
for pid in data['G'].keys():
    if pid == "159" or pid == "238" or pid == "398" or pid == "415" or pid == "421":
        continue
    all_pids.append(pid)

if generate_gated_train_dev_test_set:
    # reshuffle
    print (f"Splitting gated data set into train/dev/test as {train_set_size}/{dev_set_size}/{test_set_size}")
    random.shuffle(all_pids)
    total_pids = len(all_pids)
    tidx = int(total_pids*train_set_size)
    didx = int(total_pids*(train_set_size + dev_set_size))
    train_pids = all_pids[0:tidx]
    dev_pids = all_pids[tidx:didx]
    test_pids = all_pids[didx:]

    # dump into train/dev file
    fname = odir + "/gated_train_dev_pids.dump"
    with open(fname, 'wb') as fout:
        print (f"Saving train/dev into {fname}")
        pickle.dump((train_pids, dev_pids), fout, protocol=4)

    fname = odir + "/gated_test_pids.dump"
    with open(fname, 'wb') as fout:
        print(f"Saving train/dev into {fname}")
        pickle.dump(test_pids, fout, protocol=4)

    # print stats
    for pids, name in zip((train_pids, dev_pids, test_pids), ("train", "dev", "test")):
        num_positive_samples = 0
        num_positive_slices = 0
        num_total_slices = 0
        for pid in pids:
            num_total_slices += len(data["G"][pid]['images'])
            if ('mdata' in data["G"][pid]):
                if len(data["G"][pid].keys()) > 0:
                    num_positive_samples += 1
                for key in data["G"][pid].keys():
                    num_positive_slices += 1
        print(f"{name} statistics: ")
        print(f"  Number of patients = {len(pids)}")
        print(f"  Positive samples = {num_positive_samples}")
        print(f"  Total slices = {num_total_slices}")
        print(f"  Total positive slices = {num_positive_slices}")