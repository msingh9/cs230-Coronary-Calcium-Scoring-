# Libraries
import os
import pickle
import tensorflow as tf
import random
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import shutil
from dataGenerator import dataGenerator


#random.seed(1223143)

# base class for all models
# Describe each method (function)
class BaseModel:
    def __init__(self, params=None):
        self.params = {}
        self.params['resetHistory'] = False   ;# if set to true, reset models history and start from new model
        self.params['models_dir'] = "."       ;# model directory where model's architecture is saved
        self.params['print_summary'] = True   ;# if set, print the summary of the model
        self.params['data_aug_enable'] = False ;# To enable data augmentation
        self.patience = 10                    ;# If model's val acc is not increasing after 10 epocs, terminate the training
        self.is_train = True                  ;# Set to do training

        # data placeholders
        self.train_pids = None
        self.dev_pids = None
        self.test_pids = None

        # setting parameters based on the params as given by the user
        if params is not None:
            for key, value in params.items():
                self.params[key] = value

        # combining model directory and name of model
        self.name = self.params['models_dir'] + '/' + self.name

        # create model directory if unknown
        if not os.path.isdir(self.params['models_dir']):
            os.makedirs(self.params['models_dir'])

        # Copy the train.py
        if self.is_train:
            shutil.copyfile('train.py', self.params['models_dir'] + '/train.py.copy')

        # load model if model is already there
        print (self.name)
        if not self.params['resetHistory'] and os.path.isfile(self.name + '.h5'):
            print("Loading model from " + self.name + '.h5')
            self.model = load_model(self.name + '.h5', custom_objects={'loss' : self.loss, 'seg_f1': self.seg_f1, 'class_acc': self.class_acc})
            if self.history:
                with open(self.name + '.aux_data', 'rb') as fin:
                    self.history.train_losses, self.history.val_losses, \
                    self.history.train_seg_f1, self.history.val_seg_f1, \
                    self.history.train_class_acc, self.history.val_class_acc = pickle.load(fin)
                print (self.history.train_losses)

        # Check if model is defined
        if not self.model:
            exit("model not defined")

        # print summary if user intends to
        if self.params['print_summary']:
            print(self.model.summary())

    def save(self, name=None):
        self.sfname = self.name
        if name:
            self.sfname = name
        self.model.save(self.sfname + '.h5')
        with open(self.sfname + '.aux_data', 'wb') as fout:
            pickle.dump((self.history.train_losses, self.history.val_losses,
                         self.history.train_seg_f1, self.history.val_seg_f1,
                         self.history.train_class_acc, self.history.val_class_acc
                         ), fout)
        if not name:
            pass
            #plot_model(self.model, to_file=self.name + '.png')

    # Loss function
    def loss(self, targets, inputs, smooth=1e-6):
        # reference: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        # flatten label and prediction tensors
        inputs_s = K.flatten(inputs[:, :, :, 0])
        targets_s = K.flatten(targets[:, :, :, 0])

        intersection = K.sum(targets_s * inputs_s)
        dice = (2 * intersection + smooth) / (K.sum(targets_s) + K.sum(inputs_s) + smooth)
        dice_loss = 1 - dice

        cce = tf.keras.losses.CategoricalCrossentropy()
        # Include CE loss when target is 1.
        y_true = targets[:, :, :, 1:]
        y_pred = inputs[:, :, :, 1:]
        mask = targets[:, :, :, 0]
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        cross_entropy_loss = cce(y_true_masked, y_pred_masked)
        return self.params['alpha'] * dice_loss + (1-self.params['alpha']) * cross_entropy_loss

    # Our own evaluation metric
    def seg_f1(self, y_true, y_pred):
        y_true = y_true[:, :, :, 0]
        y_pred = K.cast(y_pred[:, :, :, 0] > 0.5, tf.float32)
        tp = K.sum(y_true * y_pred)
        tn = K.sum((1 - y_true) * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
        f1 = (2 * p * r) / (p + r + K.epsilon())
        return f1

    def class_acc(self, y_true, y_pred):
        mask = y_true[:, :, :, 0]
        y_true_masked = tf.boolean_mask(y_true[:, :, :, 1:], mask)
        y_pred_masked = tf.boolean_mask(y_pred[:, :, :, 1:], mask)
        if tf.equal(tf.size(y_true_masked), 0):
            return 1.0
        cclasses = tf.math.argmax(y_true_masked, axis=-1)
        pclasses = tf.math.argmax(y_pred_masked, axis=-1)
        correct_predictions = tf.math.equal(cclasses, pclasses)
        accuracy = tf.math.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy


    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=[self.seg_f1, self.class_acc])

    def train(self, batch_size, epochs, lr_scheduler):
        # default
        self.model.fit(dataGenerator(self.train_pids, batch_size, upsample_ps=self.params['upsample_ps'],
                                     limit_pids=self.params['limit_pids'], ddir=self.params['ddir']),
                       batch_size=batch_size, epochs=epochs,
                       validation_data=dataGenerator(self.dev_pids, batch_size, ddir=self.params['ddir']),
                       callbacks = [self.history,
                                        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)])

    def train_plot(self, fig=None, ax=None, show_plot=True, label=None):
        if not label:
            label = self.name
        if not fig:
            fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].plot(self.history.train_losses[self.history.acc_epochs:], label=label + ' train', color='red')
        ax[0].plot(self.history.val_losses[self.history.acc_epochs:], label=label +' val', color='blue')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epocs')
        ax[0].set_title("Loss vs epocs, train(Red)")

        ax[1].plot(self.history.train_seg_f1[self.history.acc_epochs:], label=label + ' train f1', color='red')
        ax[1].plot(self.history.val_seg_f1[self.history.acc_epochs:], label=label + ' val f1', color='blue')
        ax[1].set_ylabel('Segmentation F1')
        ax[1].set_xlabel('epocs')
        ax[1].set_title("F1 vs epocs, train(Red)")

        ax[2].plot(self.history.train_class_acc[self.history.acc_epochs:], label=label + ' class acc', color='red')
        ax[2].plot(self.history.val_class_acc[self.history.acc_epochs:], label=label + ' val class acc', color='blue')
        ax[2].set_ylabel('Class accuracy')
        ax[2].set_xlabel('epocs')
        ax[2].set_title("Class acc vs epocs, train(Red)")


        print('train_loss: ' + str(self.history.train_losses[-5:-1]))
        print('val_loss: ' + str(self.history.val_losses[-5:-1]))
        print('seg_f1: ' + str(self.history.train_seg_f1[-5:-1]))
        print('val_seg_f1: ' + str(self.history.val_seg_f1[-5:-1]))
        print('seg_class_acc: ' + str(self.history.train_class_acc[-5:-1]))
        print('val_class_acc: ' + str(self.history.val_class_acc[-5:-1]))
        print('epochs:   ' + str(len(self.history.train_losses)))

        if show_plot:
            plt.show()

    # over write predict method
    def my_predict(self, pids, batch_size):
        return self.model.predict(dataGenerator(pids, batch_size, shuffle=False), batch_size)
