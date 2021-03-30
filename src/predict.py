from omegaconf import OmegaConf
import sys

sys.path.append("./models/")
import utils

# from wavenet import WaveNet
# from resnet_1 import ResNet_1
import wavenet
import resnet_1
import resnet_2
import lstm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy as np

from model import BaseModel

MODEL_NAMES_DICT = {
    'wavenet': wavenet.Model,
    "resnet_1": resnet_1.Model,
    "resnet_2": resnet_2.Model,
    "lstm": lstm.Model
}


def main(param):
    utils.seed_everything(0)
    print('read csv...')
    train, test, submit = utils.read_data("./data", pseudo=True)
    print('read wave data...')
    train_wave = utils.read_wave("./data/ecg/" + train["Id"] + ".npy")
    test_wave = utils.read_wave("./data/ecg/" + test["Id"] + ".npy")
    train_y = train["target"]

    train["sex"] = train["sex"].replace({"male": 0, "female": 1})
    test["sex"] = test["sex"].replace({"male": 0, "female": 1})

    human_mask = train['label_type'] == 'human'

    train_meta_human = train[human_mask][["sex", "age"]]
    train_wave_human = train_wave[human_mask]

    train_meta_auto = train[~human_mask][["sex", "age"]]
    train_wave_auto = train_wave[~human_mask]

    train_y_human = train_y[human_mask]
    train_y_auto = train_y[~human_mask]

    kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    test_preds = np.zeros(
        [
            5,
            test_wave.shape[0]
        ]
    )

    for (fold, (train_index, val_index)) in enumerate(kf.split(train_meta_human, train_y_human)):
        print(f"{'=' * 20} fold {fold + 1} {'=' * 20}")

        # foldごとに定義しないとリークしてしまう
        model = MODEL_NAMES_DICT[param.model_name](param)

        test_preds[fold] = model.predict([test_wave, test[["sex", "age"]]], fold)

    submit["target"] = test_preds.mean(axis=0)
    submit.to_csv("./logs/{}/submission.csv".format(param.model_name), index=False)


if __name__ == '__main__':
    param = OmegaConf.from_cli()
    print('params:', param)
    main(param)
