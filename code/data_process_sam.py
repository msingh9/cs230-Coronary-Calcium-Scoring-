# def transform_to_hu(slices):
#     images = np.stack([file.pixel_array for file in slices])
#     images = images.astype(np.int16)
#     images = set_outside_scanner_to_air(images)
#
#     # convert to HU
#     for n in range(len(slices)):
#
#         intercept = slices[n].RescaleIntercept
#         slope = slices[n].RescaleSlope
#
#         if slope != 1:
#             images[n] = slope * images[n].astype(np.float64)
#             images[n] = images[n].astype(np.int16)
#
#         images[n] += np.int16(intercept)
#
#     return np.array(images, dtype=np.int16)
#
# hu_images = transform_to_hu(images)



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
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# set the random seed to create the same train/val/test split
np.random.seed(10015321)
debug = 2

# for debug messages
def myprint(msg, level):
    if level <= debug:
        print (msg)

print("test")

# data path directory
ddir = "/Users/namaste/PycharmProjects/cs230data/dataset/cocacoronarycalciumandchestcts-2/"
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

doPlot = True
plot3D = True
images = []
pid = 81 # specify the patient id
sdir_id = "G" ;# G for gated, N for non-gated


# cornary name to id
cornary_name_2_id = {"Right Coronary Artery": 0,
                     "Left Anterior Descending Artery": 1,
                     "Left Coronary Artery": 2,
                     "Left Circumflex Artery":3}

# pixel colors
pixel_colors = {0: 'red',
                1: 'blue',
                2: 'green',
                3: 'yellow'}

# get pixel coordinates
def get_pix_coords(pxy):
    x, y = eval(pxy)
    x = int(x+0.99)
    y = int(y+0.99)
    assert x > 0 and x < 512, f"Invalid {x} value for pixel coordinate"
    assert y > 0 and y < 512, f"Invalid {y} value for pixel coordinate"
    return (x, y)

# process calcium_xml
# return data is:
#  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
def process_xml(f):
    # input XML file
    # output - directory containing various meta data
    with open(f, 'rb') as fin:
        pl = plistlib.load(fin)
    # extract needed info from XML
    data = {}
    for image in pl["Images"]:
        iidx = image["ImageIndex"]
        num_rois = image["NumberOfROIs"]
        assert num_rois == len(image["ROIs"]), f"{num_rois} ROIs but not all specified in {f}"
        for roi in image["ROIs"]:
            if (len(roi['Point_px']) > 0):
                if iidx not in data:
                    data[iidx] = []
                data[iidx].append({"cid" : cornary_name_2_id[roi['Name']]})
                assert len(roi['Point_px']) == roi['NumberOfPoints'], f"Number of ROI points does not match with given length for {f}"
                data[iidx][-1]['pixels'] = [get_pix_coords(pxy) for pxy in roi['Point_px']]
            else:
                print (f"Warning: ROI without pixels specified for {iidx} in {f}")

    return data


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
                images.append(ds)


    def set_outside_scanner_to_air(raw_pixelarrays):
        # in OSIC we find outside-scanner-regions with raw-values of -2000.
        # Let's threshold between air (0) and this default (-2000) using -1000
        raw_pixelarrays[raw_pixelarrays <= -1000] = 0
        print("test")
        return raw_pixelarrays

    def transform_to_hu(slices):
        images = np.stack([file.pixel_array for file in slices])
        images = images.astype(np.int16)
        images = set_outside_scanner_to_air(images)

        # convert to HU
        for n in range(len(slices)):

            intercept = slices[n].RescaleIntercept
            slope = slices[n].RescaleSlope

            if slope != 1:
                images[n] = slope * images[n].astype(np.float64)
                images[n] = images[n].astype(np.int16)

            images[n] += np.int16(intercept)

        return np.array(images, dtype=np.int16)

    hu_images = transform_to_hu(images)


    plt.imshow(images[4].pixel_array, plt.cm.bone)
    print("printing non-HU version")
    print(images[4].pixel_array[300, 300])
    plt.show()

    plt.imshow(hu_images[4], plt.cm.bone)
    print("printing HU version")
    print(hu_images[4][300, 300])
    plt.show()


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
                if len(data[k][pid].keys() > 0):
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