import tensorflow as tf
import tensorflow_addons as tfa
import utils
from model import BaseModel
from tensorflow.keras import losses
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling1D, Add, Activation
from tensorflow.keras.optimizers import Adam


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


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params, get_model())
