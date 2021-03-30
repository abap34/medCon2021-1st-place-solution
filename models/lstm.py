import tensorflow as tf
import tensorflow_addons as tfa
import utils
from tensorflow.keras import losses
from tensorflow.keras.layers import Activation  # , Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from model import BaseModel


def get_model(input_shape=(800, 12)):
    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=2)  # 入力の形状の指定. shape=(時間軸, 12誘導)
    # block1

    x = Conv1D(filters=32, kernel_size=5, strides=1)(input1)
    x = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    x = Activation("relu")(x)
    x = Dense(32)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(16)(x)

    x = GlobalAveragePooling1D()(x)

    y = tf.keras.layers.Dense(2)(input2)
    y = tf.keras.layers.Activation("relu")(y)
    # model1 = Model(inputs=input1, outputs=x)
    # model2 = Model(inputs=input2, outputs=y)

    combined = tf.keras.layers.concatenate(
        [
            x,
            y
        ]
    )
    z = Dense(10, activation="tanh")(combined)
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


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params, get_model())
