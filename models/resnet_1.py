import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling1D, Add, Activation  # , Dropout

from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.op_def_library import value_to_attr_value
import tensorflow_addons as tfa
from tensorflow.keras import losses
from tensorflow.keras.layers import GlobalAveragePooling1D

import matplotlib.pyplot as plt
import pandas as pd

import os
from omegaconf import OmegaConf

import utils


def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):
    def build_residual_block(l_input):
        resid_input = l_input
        for dilation_rate in [2 ** i for i in range(stacked_layer)]:
            l_sigmoid_conv1d = Conv1D(
                num_filters, kernel_size, dilation_rate=dilation_rate,
                padding='same', activation='sigmoid')(l_input)
            l_tanh_conv1d = Conv1D(
                num_filters, kernel_size, dilation_rate=dilation_rate,
                padding='same', activation='relu')(l_input)
            l_input = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
            l_input = Conv1D(num_filters, 1, padding='same')(l_input)
            resid_input = Add()([resid_input, l_input])
        return resid_input

    return build_residual_block


def get_model(input_shape=(800, 12)):
    num_filters_ = 16
    kernel_size_ = 4
    stacked_layers_ = [12, 8, 4, 1]
    l_input = Input(shape=(input_shape))
    cat_input = Input(shape=(2,), name='cat_input')
    x = Conv1D(num_filters_, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_, kernel_size_, stacked_layers_[0])(x)
    x = Conv1D(num_filters_ * 2, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_ * 2, kernel_size_, stacked_layers_[1])(x)
    x = Conv1D(num_filters_ * 4, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_ * 4, kernel_size_, stacked_layers_[2])(x)
    x = Conv1D(num_filters_ * 8, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_ * 8, kernel_size_, stacked_layers_[3])(x)
    x = GlobalAveragePooling1D()(x)
    l_output = Dense(32, activation='relu')(x)
    concat_layer_out = tf.keras.layers.Concatenate()([l_output, cat_input])
    model_output = Dense(1, activation='sigmoid')(concat_layer_out)
    model = models.Model(inputs=[l_input, cat_input], outputs=[model_output])
    opt = Adam(lr=1e-3)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=losses.BinaryCrossentropy(label_smoothing=0.001), metrics=['AUC'], optimizer=opt)
    return model


def get_model(input_shape=(800, 12)):
    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=2)  # 入力の形状の指定. shape=(時間軸, 12誘導)
    # block1

    C = Conv1D(filters=32, kernel_size=5, strides=1)(input1)

    C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A11)
    S11 = Add()([C12, C])
    A12 = Activation("relu")(S11)
    M11 = MaxPooling1D(pool_size=5, strides=2)(A12)

    C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M11)
    A21 = Activation("relu")(C21)
    C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A21)
    S21 = Add()([C22, M11])
    A22 = Activation("relu")(S11)
    M21 = MaxPooling1D(pool_size=5, strides=2)(A22)

    C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M21)
    A31 = Activation("relu")(C31)
    C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A31)
    S31 = Add()([C32, M21])
    A32 = Activation("relu")(S31)
    M31 = MaxPooling1D(pool_size=5, strides=2)(A32)

    C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M31)
    A41 = Activation("relu")(C41)
    C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A41)
    S41 = Add()([C42, M31])
    A42 = Activation("relu")(S41)
    M41 = MaxPooling1D(pool_size=5, strides=2)(A42)

    C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)
    A51 = Activation("relu")(C51)
    C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)
    S51 = Add()([C52, M41])
    A52 = Activation("relu")(S51)
    M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

    x = tf.keras.layers.GlobalAveragePooling1D()(M51)

    y = tf.keras.layers.Dense(2)(input2)
    y = tf.keras.layers.Activation("relu")(y)

    combined = tf.keras.layers.concatenate(
        [
            x,
            y
        ]
    )
    z = Dense(32, activation="tanh")(combined)
    z = Dense(1, activation="sigmoid")(z)

    model = utils.SAMModel(
        inputs=[
            input1,
            input2
        ],
        outputs=z
    )

    opt = Adam(lr=1e-3)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=losses.BinaryCrossentropy(label_smoothing=0.001), metrics=['AUC'], optimizer=opt)

    return model


class ResNet_1:
    def __init__(self, params):
        self.params = params
        # logファイルを作る
        os.makedirs('./logs/{}/models'.format(self.params.model_name), exist_ok=True)
        os.makedirs('./logs/{}/losses'.format(self.params.model_name), exist_ok=True)
        # model作る
        self.model = get_model()

    def fit(self, train_x, train_y, val_x, val_y, fold_index):
        # 学習
        # --------------------------------------------------

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./logs/{}/models/fold-{}.ckpt'.format(self.params.model_name, fold_index),
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )

        history = self.model.fit(
            train_x,
            train_y,
            epochs=self.params.epoch,  # 25
            batch_size=self.params.batch_size,  # 8
            verbose=1,
            validation_data=(
                val_x,
                val_y
            ),
            callbacks=[model_checkpoint_callback]
        )

        # --------------------------------------------------

        # log(学習曲線とloss推移)
        # --------------------------------------------------

        plt.plot(history.history["auc"])
        plt.plot(history.history["val_auc"])
        plt.title("Learning Curve")
        plt.ylabel("AUC")
        plt.xlabel("Epoch")
        plt.grid()
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("./logs/{}/losses/fold-{}.png".format(self.params.model_name, fold_index))
        plt.clf()

        learning_logs = pd.DataFrame(history.history)
        learning_logs.to_csv('./logs/{}/losses/log.csv'.format(self.params.model_name, fold_index))
        OmegaConf.save(self.params, './logs/{}/params.yaml'.format(self.params.model_name, fold_index))

        # --------------------------------------------------

        return self.model.predict(val_x)
