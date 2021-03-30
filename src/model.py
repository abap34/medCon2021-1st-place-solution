import os

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


class BaseModel:
    def __init__(self, params, model):
        self.params = params
        # logファイルを作る
        os.makedirs('./logs/{}/models'.format(self.params.model_name), exist_ok=True)
        os.makedirs('./logs/{}/losses'.format(self.params.model_name), exist_ok=True)
        # model作る
        self.model = model

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
        learning_logs.to_csv('./logs/{}/losses/fold-{}-log.csv'.format(self.params.model_name, fold_index))
        OmegaConf.save(self.params, './logs/{}/params.yaml'.format(self.params.model_name, fold_index))

        # --------------------------------------------------

        return self.model.predict(val_x)
