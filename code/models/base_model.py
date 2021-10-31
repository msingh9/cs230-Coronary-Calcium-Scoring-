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

random.seed(1223143)

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
            if self.params['poisson']:
                self.model = load_model(self.name + '.h5', custom_objects={'exp': tf.math.exp})
            else:
                self.model = load_model(self.name + '.h5')
            if self.history:
                with open(self.name + '.aux_data', 'rb') as fin:
                    self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc = pickle.load(fin)

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
            pickle.dump((self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc), fout)
        if not name:
            plot_model(self.model, to_file=self.name + '.png')

    # Loss function
    def loss(self, targets, inputs, smooth=1e-6):

        # flatten label and prediction tensors
        inputs_s = K.flatten(inputs[:,:,:,0])
        targets_s = K.flatten(targets[:,:,:,0])

        intersection = K.sum(targets_s * inputs_s)
        dice = (2 * intersection + smooth) / (K.sum(targets_s) + K.sum(inputs_s) + smooth)
        dice_loss = 1 - dice

        cce = tf.keras.losses.CategoricalCrossentropy()
        # Include CE loss when target is 1.
        cross_entropy_loss = cce(targets[:,:,:,1:] * targets[:,:,:,0:1], inputs[:,:,:,1:] * targets[:,:,:,0:1])
        return dice_loss + cross_entropy_loss

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

    def train(self, batch_size, epochs, lr_scheduler):
        # default
        self.model.fit(dataGenerator(self.train_pids, batch_size), batch_size=batch_size, epochs=epochs,
                       validation_data=(self.dev_pids),
                       callbacks = [self.history,
                                        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.patience)])

    def train_plot(self, fig=None, ax=None, show_plot=True, label=None):
        if not label:
            label = self.name
        if not fig:
            fig, ax = plt.subplots(nrows=1, ncols=2)

            ax[0].plot(self.history.train_losses[self.history.acc_epochs:], label=label + ' train', color='red')
            ax[0].plot(self.history.val_losses[self.history.acc_epochs:], label=label +' val', color='blue')
            ax[0].set_ylabel('Loss')
            ax[0].set_xlabel('epocs')
            ax[0].set_title("Loss vs epocs, train(Red)")

            ax[1].plot(self.history.train_acc[self.history.acc_epochs:], label=label + ' train', color='red')
            ax[1].plot(self.history.val_acc[self.history.acc_epochs:], label=label + ' val', color='blue')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_xlabel('epocs')
            ax[1].set_title("Accuracy vs epocs, train(Red)")

            print('train_loss: ' + str(self.history.train_losses[-5:-1]))
            print('val_loss: ' + str(self.history.val_losses[-5:-1]))
            print('train_acc: ' + str(self.history.train_acc[-5:-1]))
            print('val_acc: ' + str(self.history.val_acc[-5:-1]))
            print('epochs:   ' + str(len(self.history.train_losses)))

        plot_file_name = self.name + "_plot.png"
        print (f"Saving plot in {plot_file_name}")
        plt.savefig(plot_file_name)

        if show_plot:
            plt.show()

    # over write predict method
    def my_predict(self, x, batchsize):
        return self.model.predict(x, batchsize)
