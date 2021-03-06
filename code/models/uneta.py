import os
#import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import concatenate, add, multiply, Lambda
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.keras.models import Sequential
from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Model as m
from tensorflow.keras import backend
import tensorflow.keras.backend as K

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'uneta'
        self.model = Sequential()
        self.batchnorm = False
        self.dropout = 0.0

        inputs = Input(shape=(512, 512, 1))


        # From https://github.com/nabsabraham/focal-tversky-unet

        # K.set_image_data_format('channels_last')  # TF dimension ordering in this code
        kinit = 'glorot_normal'

        def expend_as(tensor, rep,name):
            my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
            return my_repeat

        # plain old attention gates in u-net, NO multi-input, NO deep supervision
        def UnetConv2D(input, outdim, is_batchnorm, name):
            x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
            if is_batchnorm:
                x =BatchNormalization(name=name + '_1_bn')(x)
            x = Activation('relu',name=name + '_1_act')(x)

            x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
            if is_batchnorm:
                x = BatchNormalization(name=name + '_2_bn')(x)
            x = Activation('relu', name=name + '_2_act')(x)
            return x

        def UnetGatingSignal(input, is_batchnorm, name):
            ''' this is simply 1x1 convolution, bn, activation '''
            shape = K.int_shape(input)
            x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
            if is_batchnorm:
                x = BatchNormalization(name=name + '_bn')(x)
            x = Activation('relu', name = name + '_act')(x)
            return x


        def AttnGatingBlock(x, g, inter_shape, name, dropout_block):
            ''' take g which is the spatially smaller signal, do a conv to get the same
            number of feature channels as x (bigger spatially)
            do a conv on x to also get same geature channels (theta_x)
            then, upsample g to be same size as x
            add x and g (concat_xg)
            relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''

            shape_x = K.int_shape(x)  # 32
            shape_g = K.int_shape(g)  # 16

            theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16
            shape_theta_x = K.int_shape(theta_x)

            phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
            upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

            concat_xg = add([upsample_g, theta_x])
            act_xg = Activation('relu')(concat_xg)
            if dropout_block:
              act_xg = Dropout(dropout_block, name='drop_psi'+name)(act_xg)
            psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
            sigmoid_xg = Activation('sigmoid')(psi)
            shape_sigmoid = K.int_shape(sigmoid_xg)
            upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

            upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
            y = multiply([upsample_psi, x], name='q_attn'+name)

            result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
            result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
            return result_bn

        def get_attn_unet(dropout_rate, dropout_block, input_size=(512, 512, 1)):
            # New: adds dropout to the attention gating block before the linear layer.
            inputs = Input(shape=input_size)
            conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
            if dropout_rate:
              conv3 = Dropout(dropout_rate,name='drop_conv3')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
            if dropout_rate:
              conv4 = Dropout(dropout_rate, name='drop_conv4')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

            g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
            attn1 = AttnGatingBlock(conv4, g1, 128, '_1', dropout_block)
            up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')

            g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
            attn2 = AttnGatingBlock(conv3, g2, 64, '_2', dropout_block)
            up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

            g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
            attn3 = AttnGatingBlock(conv2, g3, 32, '_3', dropout_block)
            up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

            up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
            out = Conv2D(1, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)
            model = m(inputs=[inputs], outputs=out)

            return model

            # Model
        self.model = get_attn_unet(0.2, dropout_block=0.2, input_size=(512, 512, 1))

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)