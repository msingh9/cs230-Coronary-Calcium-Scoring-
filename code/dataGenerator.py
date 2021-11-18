from pydicom import dcmread
from my_lib import process_xml
import tensorflow as tf
import math
import os
import numpy as np
from matplotlib.path import Path
from tensorflow.keras.utils import to_categorical
import sys
import random

class dataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pids, batch_size, ddir="../dataset",
                 upsample_ps=0, limit_pids=None, shuffle=True,
                 only_use_pos_images=False, data_aug_enable=False):
        if limit_pids:
            self.pids = pids[0:limit_pids]
        else:
            self.pids = pids
        print (ddir)
        if (ddir == "../mini_dataset"):
            self.ddir = ddir + "/Gated_release_final"
        else:
            self.ddir = ddir + "/cocacoronarycalciumandchestcts-2/Gated_release_final"
        # Load all the images across pids
        self.X = []
        self.mdata = []
        self.upsample_ps = upsample_ps
        self.cache = {}
        self.shuffle = shuffle
        self.only_use_pos_images = only_use_pos_images
        self.data_aug_enable = data_aug_enable

        # Estimate total work
        total_work = 0
        progress_count = 0
        for pid in self.pids:
            for subdir, dirs, files in os.walk(self.ddir + "/patient/" + str(pid) + '/'):
                total_work += len(files)

        print ("Loading dataset")
        for i, pid in enumerate(self.pids):
            for subdir, dirs, files in os.walk(self.ddir + "/patient/" + str(pid) + '/'):
                for iidx, filename in enumerate(sorted(files, reverse=True)):
                    progress_count += 1
                    if (progress_count % 64 == 0):
                        p = int(progress_count * 100 / total_work)
                        sys.stdout.write(f"\r{p}%")
                        sys.stdout.flush()
                    filepath = subdir + os.sep + filename
                    if filepath.endswith(".dcm"):
                        self.X.append(dcmread(filepath).pixel_array)
                        self.mdata.append((pid, iidx)) ;# iidx is image index
        self.batch_size = batch_size
        sys.stdout.write("\n")

        if self.only_use_pos_images:
            print(f"Filtering to only have positive images")
            new_X = []
            new_mdata = []
            for index, (pid, iidx) in enumerate(self.mdata):
                fname = self.ddir + "/calcium_xml/" + str(pid) + (".xml")
                if not os.path.exists(fname):
                    continue
                if fname not in self.cache:
                    mdata = process_xml(fname)
                    self.cache[fname] = mdata
                else:
                    mdata = self.cache[fname]
                # mdata format is:
                #  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
                if iidx not in mdata:
                    continue
                new_X.append(self.X[index])
                new_mdata.append((pid, iidx))
            self.X = new_X
            self.mdata = new_mdata

        # Normalize Xs
        print(f'Read {len(self.X)} examples before upsampling.')
        print(f"Image pixel data type before normalization is {self.X[0][0, 0].dtype} {self.X[0][0, 0]}")
        norm_const = np.array(2 ** 16 - 1).astype('float32')
        print("Normalizing inputs, it takes a little while")
        self.X = self.X / norm_const
        print(f"Image pixel data type after normalization {self.X[0][0, 0].dtype} {self.X[0][0, 0]}")

        # Up sample image
        if self.upsample_ps:
            print(f"Upsampling positive samples by {self.upsample_ps}")
            new_X = []
            new_mdata = []
            for index, (pid, iidx) in enumerate(self.mdata):
                fname = self.ddir + "/calcium_xml/" + str(pid) + (".xml")
                new_X.append(self.X[index])
                new_mdata.append((pid, iidx))
                if not os.path.exists(fname):
                    continue
                if fname not in self.cache:
                    mdata = process_xml(fname)
                    self.cache[fname] = mdata
                else:
                    mdata = self.cache[fname]
                # mdata format is:
                #  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
                if iidx not in mdata:
                    continue
                for i in range(self.upsample_ps):
                    new_X.append(self.X[index])
                    new_mdata.append((pid, iidx))
            self.X = new_X
            self.mdata = new_mdata
        print(f'Using {len(self.X)} examples after optional upsampling.')
        # Reshuffle
        if self.shuffle:
            indices = [i for i in range(len(self.X))]
            random.shuffle(indices)
            xxx = [self.X[i] for i in indices]
            yyy = [self.mdata[i] for i in indices]
            self.X = xxx
            self.mdata = yyy

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        if ((idx +1 ) * self.batch_size) > len(self.X):
            Xs = self.X[idx * self.batch_size : len(self.X)]
            mdatas = self.mdata[idx * self.batch_size : len(self.X)]
        else:
            Xs = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            mdatas = self.mdata[idx * self.batch_size:(idx + 1) * self.batch_size]

        height, width = Xs[0].shape
        m = len(Xs)
        Ys = np.zeros((m, height, width, 1))

        # load XML and prepare Ys
        for index, (pid, iidx) in enumerate(mdatas):
            fname = self.ddir + "/calcium_xml/" + str(pid) + (".xml")
            if not os.path.exists(fname):
                continue
            if fname not in self.cache:
                mdata = process_xml(fname)
                self.cache[fname] = mdata
            else:
                mdata = self.cache[fname]
            # mdata format is:
            #  {<image_index>: [{cid: <integer>, pixels: [(x1,y1), (x2,y2)..]},..]
            if iidx not in mdata:
                continue
            for _ in mdata[iidx]:
                poly_path = Path(_['pixels'])
                y, x = np.mgrid[:height, : width]
                coors = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
                mask = poly_path.contains_points(coors).reshape(height, width)
                Ys[index, :, :, 0] += mask
                #Ys[index, :, :, 1] = (Ys[index, :, :, 1] * ~mask) + (np.full((height, width), _['cid']) * mask)

        #Ys[:,:,:,1:5] = to_categorical(Ys[:, :, :, 1], num_classes=4)
        return np.array(Xs).reshape(m, height, width, 1), Ys

    def on_epoch_end(self):
        # Reshuffle
        if self.shuffle:
            indices = [i for i in range(len(self.X))]
            random.shuffle(indices)
            xxx = [self.X[i] for i in indices]
            yyy = [self.mdata[i] for i in indices]
            self.X = xxx
            self.mdata = yyy
