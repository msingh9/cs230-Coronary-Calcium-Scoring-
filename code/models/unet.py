import os
#import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Model as m
from tensorflow.keras import backend

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'unet'
        self.model = Sequential()
        self.batchnorm = True
        self.dropout = 0.8

        inputs = Input(shape=(512, 512, 1))

        def conv2d_block(input_tensor, n_filters, kernel_size=3):
            """Function to add 2 convolutional layers with the parameters passed to it"""
            # first layer
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                       kernel_initializer='he_normal', padding='same')(input_tensor)
            if self.batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # second layer
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                       kernel_initializer='he_normal', padding='same')(x)
            if self.batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            return x

        # Encoder path
        # Conv2D-64 (3x3, same) -> 512x512x64 -> maxpool -> 256x256x64
        c1 = conv2d_block(inputs, 64)
        p1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c1)
        if self.dropout:
            p1 = Dropout(self.dropout)(p1)

        # Conv2D-128 (3x3, same) -> 256x256x128 -> maxpool -> 128x128x128
        c2 = conv2d_block(p1, 128)
        p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c2)
        if self.dropout:
            p2 = Dropout(self.dropout)(p2)

        # Conv2D-256 (3x3, same) -> 128x128x128 -> maxpool -> 64x64x256
        c3 = conv2d_block(p2, 256)
        p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c3)
        if self.dropout:
            p3 = Dropout(self.dropout)(p3)

        # Conv2D-512 (3x3, same) -> 64x64x256 -> maxpool -> 32x32x512
        c4 = conv2d_block(p3, 512)
        p4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c4)
        if self.dropout:
            p4 = Dropout(self.dropout)(p4)

        # Conv2D-512 (3x3, same) -> 32x32x1024
        c5 = conv2d_block(p4, 1024)
        p5 = c5
        if self.dropout:
            p5 = Dropout(self.dropout)(p5)

        # Decoder path (64x64)
        u4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(p5)
        u4 = concatenate([u4, c4])
        if self.dropout:
            u4 = Dropout(self.dropout)(u4)
        u4 = conv2d_block(u4, 512)

        # 128x128
        u3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(u4)
        u3 = concatenate([u3, c3])
        if self.dropout:
            u3 = Dropout(self.dropout)(u3)
        u3 = conv2d_block(u3, 256)

        # 256x256
        u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u3)
        u2 = concatenate([u2, c2])
        if self.dropout:
            u2 = Dropout(self.dropout)(u2)
        u2 = conv2d_block(u2, 128)

        # 512x512
        u1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(u2)
        u1 = concatenate([u1, c1])
        if self.dropout:
            u1 = Dropout(self.dropout)(u1)
        u1 = conv2d_block(u1, 64)

        # Final layer
        output0 = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(u1)
        output1 = Conv2D(4, kernel_size=(1, 1), activation="softmax")(u1)
        outputs = concatenate([output0, output1])

        # Model
        self.model = m(inputs=[inputs], outputs=outputs)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)