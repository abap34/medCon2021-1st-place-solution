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

def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):
    def build_residual_block(l_input):
        resid_input = l_input
        for dilation_rate in [2**i for i in range(stacked_layer)]:
            l_sigmoid_conv1d = Conv1D(
              num_filters, kernel_size, dilation_rate=dilation_rate,
              padding='same', activation='sigmoid')(l_input)
            l_tanh_conv1d = Conv1D(
             num_filters, kernel_size, dilation_rate=dilation_rate,
             padding='same', activation='relu')(l_input)
            l_input = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
            l_input = Conv1D(num_filters, 1, padding='same')(l_input)
            resid_input = Add()([resid_input ,l_input])
        return resid_input
    return build_residual_block


def get_model(input_shape=(800, 12)):
    num_filters_ = 16
    kernel_size_ = 4
    stacked_layers_ = [12, 8, 4, 1]
    l_input = Input(shape=(input_shape))
    cat_input = Input(shape=(2, ), name='cat_input')
    x = Conv1D(num_filters_, 1, padding='same')(l_input)
    x = WaveNetResidualConv1D(num_filters_, kernel_size_, stacked_layers_[0])(x)
    x = Conv1D(num_filters_*2, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_*2, kernel_size_, stacked_layers_[1])(x)
    x = Conv1D(num_filters_*4, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_*4, kernel_size_, stacked_layers_[2])(x)
    x = Conv1D(num_filters_*8, 1, padding='same')(x)
    x = WaveNetResidualConv1D(num_filters_*8, kernel_size_, stacked_layers_[3])(x)
    x = GlobalAveragePooling1D()(x)
    l_output = Dense(32, activation='relu')(x)
    concat_layer_out = tf.keras.layers.Concatenate()([l_output, cat_input])
    model_output = Dense(1, activation='sigmoid')(concat_layer_out)
    model = models.Model(inputs=[l_input, cat_input], outputs=[model_output])
    opt = Adam(lr=1e-3)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=losses.BinaryCrossentropy(label_smoothing=0.001), metrics=['AUC'], optimizer=opt)
    return model

class WaveNet:
    def __init__(self,  params):
        self.params = params
        # logファイルを作る
        os.makedirs('../logs/{}/models'.format(self.params.model_name), exist_ok=True)
        os.makedirs('../logs/{}/losses'.format(self.params.model_name), exist_ok=True)
        # model作る
        self.model = get_model()

    def fit(self, train_x, train_y, val_x, val_y, fold_index):
        

        # 学習
        # --------------------------------------------------

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='../logs/{}/models/fold-{}.ckpt'.format(self.params.model_name, fold_index),  
            save_weights_only=True,  
            monitor='val_auc',  
            mode='max', 
            save_best_only=True  
        )
        
        history = self.model.fit(
            train_x, 
            train_y,
            epochs=self.params.epoch, #25
            batch_size=self.params.batch_size, #8
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
        plt.savefig("../logs/{}/losses/fold-{}".format(self.params.model_name, fold_index))
        plt.clf()

        learning_logs = pd.DataFrame(history.history)
        learning_logs.to_csv('../logs/losses/log.csv')
        OmegaConf.save(self.params, '../logs/{}/params.yaml')

        # --------------------------------------------------
        

        return self.model.predict(val_x)


