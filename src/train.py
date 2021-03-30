from omegaconf import OmegaConf
import sys
sys.path.append("./models/")
import utils

from wavenet import WaveNet

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy as np

MODEL_NAMES_DICT = {
    'wavenet':WaveNet
}

def main(param):
    print('read csv...')
    train, test, submit = utils.read_data("./data")
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
    
    model = MODEL_NAMES_DICT[param.model_name](param)

    kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    
    val_preds = np.zeros(train.shape[0])

    for (fold, (train_index, val_index)) in enumerate(kf.split(train_meta_human, train_y_human)):
        train_input_wave = np.concatenate([
            train_wave_human[train_index],
            train_wave_auto
        ])

        train_input_meta = np.concatenate([
            train_meta_human.iloc[train_index],
            train_meta_auto
        ])

        train_y_concat = np.concatenate([
            train_y_human.iloc[train_index],
            train_y_auto
        ])
        
        val_input_wave = train_wave_human[val_index]

        val_input_meta = train_meta_human.iloc[val_index]

        val_y_concat = np.concatenate([
            train_y_human.iloc[val_index],
            train_y_auto
        ])

        val_pred = model.fit(
            [train_input_wave, train_input_meta],
            train_y_concat,
            [val_input_wave, val_input_meta],
            val_y_concat,
            fold 
        )[:, 0]

        # foldを忘れないよう注意. fitの帰り値はval_pred

        val_preds[val_index] += val_pred

    print("AUC score:", roc_auc_score(train_y, val_preds))


if __name__ == '__main__':
    param = OmegaConf.from_cli()
    print('params:', param)
    main(param)